"""
春日部駅周辺を東武スカイツリー線路で2領域に分割し、
各領域からランダムポイントを生成して住所を逆ジオコーディングするプログラム

改善点:
- 緯度経度の度単位ではなく、メートル単位の投影座標で面積・距離・分割を処理
- OSMから取得した線路のうち、指定した線路区間に近いものを優先採用
- 線路がポリゴンを完全に分割しない場合は、指定端点を延長した線で安定的に分割
- Nominatimを1秒以上の間隔で呼び出し、User-Agentとキャッシュを使用
- locationがNoneの場合やMultiPolygonの場合にも対応
"""

import json
import math
import random
import time
from pathlib import Path

import geopandas as gpd
import osmnx as ox
import pandas as pd
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim
from shapely.geometry import LineString, MultiLineString, Point, Polygon
from shapely.ops import linemerge, split, transform, unary_union


# ==================== 設定パラメータ ====================

KASUKABE_STATION_LAT = 35.9749
KASUKABE_STATION_LON = 139.7474
RADIUS_M = 1000

# 線路の端点: (lat, lon)
RAILWAY_START = (35.9691, 139.7367)
RAILWAY_END = (35.9751, 139.7450)

NUM_ITERATIONS = 10
RANDOM_SEED = None  # 再現性が必要なら 42 などにする

OUTPUT_JSON = "kasukabe_random_points_results_10.json"
OUTPUT_CSV = "kasukabe_random_points_results_10.csv"

# 春日部周辺なら UTM 54N で十分実用的にメートル処理できる
WGS84_CRS = "EPSG:4326"
PROJECTED_CRS = "EPSG:32654"

NOMINATIM_USER_AGENT = "kasukabe_random_points_analysis/1.0"


# ==================== 幾何処理 ====================

def make_point_gdf(lat, lon):
    return gpd.GeoDataFrame(
        geometry=[Point(lon, lat)],
        crs=WGS84_CRS,
    )


def create_search_area_polygon(center_lat, center_lon, radius_m):
    """
    中心点から半径 radius_m の円形ポリゴンを作成する。
    メートル単位の投影座標で buffer し、WGS84 と投影座標の両方を返す。
    """
    center_wgs = make_point_gdf(center_lat, center_lon)
    center_proj = center_wgs.to_crs(PROJECTED_CRS)

    search_polygon_proj = center_proj.geometry.iloc[0].buffer(radius_m, resolution=96)

    search_gdf_proj = gpd.GeoDataFrame(geometry=[search_polygon_proj], crs=PROJECTED_CRS)
    search_polygon_wgs = search_gdf_proj.to_crs(WGS84_CRS).geometry.iloc[0]

    return search_polygon_wgs, search_polygon_proj


def make_reference_railway_line():
    """
    指定端点を結ぶ基準線路ラインを作る。
    """
    return LineString([
        (RAILWAY_START[1], RAILWAY_START[0]),
        (RAILWAY_END[1], RAILWAY_END[0]),
    ])


def to_projected_geometry(geom):
    gdf = gpd.GeoDataFrame(geometry=[geom], crs=WGS84_CRS)
    return gdf.to_crs(PROJECTED_CRS).geometry.iloc[0]


def to_wgs84_geometry(geom):
    gdf = gpd.GeoDataFrame(geometry=[geom], crs=PROJECTED_CRS)
    return gdf.to_crs(WGS84_CRS).geometry.iloc[0]


def get_candidate_railway_from_osm(search_polygon_wgs, reference_line_wgs):
    """
    OSMから railway=rail を取得し、指定した線路区間に近い線を選ぶ。
    取れない場合は None を返す。
    """
    try:
        ox.settings.log_console = False
        tags = {"railway": "rail"}

        features = ox.features_from_polygon(search_polygon_wgs, tags)

        if features.empty:
            return None

        line_geoms = []
        for geom in features.geometry:
            if geom is None:
                continue
            if geom.geom_type in ["LineString", "MultiLineString"]:
                line_geoms.append(geom)

        if not line_geoms:
            return None

        lines_gdf = gpd.GeoDataFrame(geometry=line_geoms, crs=WGS84_CRS).to_crs(PROJECTED_CRS)

        reference_proj = to_projected_geometry(reference_line_wgs)
        reference_buffer = reference_proj.buffer(120)

        nearby = lines_gdf[lines_gdf.intersects(reference_buffer)]

        if nearby.empty:
            nearby = lines_gdf

        merged = unary_union(list(nearby.geometry))

        try:
            merged = linemerge(merged)
        except Exception:
            pass

        if merged.geom_type == "LineString":
            return to_wgs84_geometry(merged)

        if merged.geom_type == "MultiLineString":
            longest = max(merged.geoms, key=lambda g: g.length)
            return to_wgs84_geometry(longest)

        return None

    except Exception as e:
        print(f"⚠️ OSM線路取得に失敗しました。指定端点の直線を使います: {e}")
        return None


def extend_line_to_cross_polygon(line_proj, polygon_proj, extension_m=3000):
    """
    線路ラインの始点・終点方向を延長し、検索円を確実に横断する線にする。
    曲線の細部よりも、2領域へ安定分割することを優先する。
    """
    coords = list(line_proj.coords)

    if len(coords) < 2:
        raise ValueError("線路ラインの座標数が不足しています。")

    x1, y1 = coords[0]
    x2, y2 = coords[-1]

    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)

    if length == 0:
        raise ValueError("線路ラインの長さが0です。")

    ux = dx / length
    uy = dy / length

    extended = LineString([
        (x1 - ux * extension_m, y1 - uy * extension_m),
        (x2 + ux * extension_m, y2 + uy * extension_m),
    ])

    return extended


def split_area_by_railway(search_polygon_proj, railway_line_wgs):
    """
    検索エリアを線路で2領域に分割する。
    分割に失敗した場合も、指定線路方向の延長線で安定的に2分割する。
    """
    railway_line_proj = to_projected_geometry(railway_line_wgs)

    if railway_line_proj.geom_type == "MultiLineString":
        railway_line_proj = max(railway_line_proj.geoms, key=lambda g: g.length)

    split_line = extend_line_to_cross_polygon(railway_line_proj, search_polygon_proj)

    try:
        divided = split(search_polygon_proj, split_line)
        parts = [geom for geom in divided.geoms if not geom.is_empty and geom.area > 1.0]

        if len(parts) >= 2:
            parts = sorted(parts, key=lambda g: g.area, reverse=True)
            return parts[0], parts[1]

    except Exception as e:
        print(f"⚠️ 線路による分割に失敗しました。簡易分割に切り替えます: {e}")

    minx, miny, maxx, maxy = search_polygon_proj.bounds
    midy = (miny + maxy) / 2
    fallback_line = LineString([(minx - 1000, midy), (maxx + 1000, midy)])

    divided = split(search_polygon_proj, fallback_line)
    parts = [geom for geom in divided.geoms if not geom.is_empty and geom.area > 1.0]

    if len(parts) < 2:
        raise RuntimeError("検索エリアを2領域に分割できませんでした。")

    parts = sorted(parts, key=lambda g: g.area, reverse=True)
    return parts[0], parts[1]


def random_point_in_polygon(polygon_proj, max_attempts=1000):
    """
    ポリゴン内にランダムポイントを生成する。
    MultiPolygonの場合は面積最大のポリゴンを使う。
    """
    if polygon_proj.geom_type == "MultiPolygon":
        polygon_proj = max(polygon_proj.geoms, key=lambda g: g.area)

    minx, miny, maxx, maxy = polygon_proj.bounds

    for _ in range(max_attempts):
        point = Point(
            random.uniform(minx, maxx),
            random.uniform(miny, maxy),
        )
        if polygon_proj.contains(point):
            return point

    return polygon_proj.representative_point()


def projected_point_to_lat_lon(point_proj):
    point_wgs = to_wgs84_geometry(point_proj)
    return point_wgs.y, point_wgs.x


# ==================== 逆ジオコーディング ====================

class ReverseGeocoder:
    def __init__(self):
        self.geolocator = Nominatim(user_agent=NOMINATIM_USER_AGENT, timeout=10)
        self.reverse_limited = RateLimiter(
            self.geolocator.reverse,
            min_delay_seconds=1.1,
            max_retries=2,
            error_wait_seconds=2.0,
            swallow_exceptions=True,
        )
        self.cache = {}

    def reverse(self, lat, lon):
        key = f"{lat:.6f},{lon:.6f}"

        if key in self.cache:
            return self.cache[key]

        location = self.reverse_limited(
            (lat, lon),
            language="ja",
            addressdetails=True,
            zoom=18,
        )

        if location is None:
            address = key
        else:
            address = location.address or key

        self.cache[key] = address
        return address


def convert_to_googlemaps_address(lat, lon, raw_address):
    """
    Google Mapsで検索しやすい形式にする。
    住所が曖昧でも座標で確実に到達できるよう、座標を必ず含める。
    """
    if raw_address and raw_address != f"{lat:.6f},{lon:.6f}":
        return f"{raw_address} ({lat:.6f}, {lon:.6f})"

    return f"{lat:.6f}, {lon:.6f}"


# ==================== メイン処理 ====================

def main():
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    print("\n" + "=" * 80)
    print("🚉 春日部駅周辺を線路で分割し、2つの領域にランダムポイントを生成")
    print("=" * 80 + "\n")

    print("📍 初期化: 検索エリアと線路を作成中...\n")

    search_polygon_wgs, search_polygon_proj = create_search_area_polygon(
        KASUKABE_STATION_LAT,
        KASUKABE_STATION_LON,
        RADIUS_M,
    )
    print(f"✅ 検索エリア作成完了: 半径 {RADIUS_M} m")

    reference_line_wgs = make_reference_railway_line()

    railway_line_wgs = get_candidate_railway_from_osm(
        search_polygon_wgs,
        reference_line_wgs,
    )

    if railway_line_wgs is None:
        railway_line_wgs = reference_line_wgs
        print("✅ 指定端点を結ぶ線路ラインを使用")
    else:
        print("✅ OSMから線路ラインを取得")

    area1_proj, area2_proj = split_area_by_railway(search_polygon_proj, railway_line_wgs)

    print(f"✅ 領域分割完了")
    print(f"   領域1 面積: {area1_proj.area:,.1f} m²")
    print(f"   領域2 面積: {area2_proj.area:,.1f} m²\n")

    geocoder = ReverseGeocoder()
    all_results = []

    for iteration in range(NUM_ITERATIONS):
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"【実行 {iteration + 1}/{NUM_ITERATIONS}】")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

        point1_proj = random_point_in_polygon(area1_proj)
        point2_proj = random_point_in_polygon(area2_proj)

        lat1, lon1 = projected_point_to_lat_lon(point1_proj)
        lat2, lon2 = projected_point_to_lat_lon(point2_proj)

        print("   📍 領域1 の住所を取得中... ", end="", flush=True)
        raw_address1 = geocoder.reverse(lat1, lon1)
        address1 = convert_to_googlemaps_address(lat1, lon1, raw_address1)
        print(f"✅\n   {address1}\n")

        print("   📍 領域2 の住所を取得中... ", end="", flush=True)
        raw_address2 = geocoder.reverse(lat2, lon2)
        address2 = convert_to_googlemaps_address(lat2, lon2, raw_address2)
        print(f"✅\n   {address2}\n")

        result = {
            "iteration": iteration + 1,
            "area1": {
                "latitude": lat1,
                "longitude": lon1,
                "raw_address": raw_address1,
                "google_maps_address": address1,
            },
            "area2": {
                "latitude": lat2,
                "longitude": lon2,
                "raw_address": raw_address2,
                "google_maps_address": address2,
            },
        }

        all_results.append(result)

    print("\n" + "=" * 80)
    print("【 最終結果 - 10組の住所ペア 】")
    print("=" * 80 + "\n")

    for result in all_results:
        print(f"【{result['iteration']}】")
        print(f"  領域1: {result['area1']['google_maps_address']}")
        print(f"  領域2: {result['area2']['google_maps_address']}")
        print()

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    csv_data = []
    for result in all_results:
        csv_data.append({
            "実行番号": result["iteration"],
            "領域1_住所": result["area1"]["google_maps_address"],
            "領域2_住所": result["area2"]["google_maps_address"],
            "領域1_緯度": result["area1"]["latitude"],
            "領域1_経度": result["area1"]["longitude"],
            "領域2_緯度": result["area2"]["latitude"],
            "領域2_経度": result["area2"]["longitude"],
        })

    df_csv = pd.DataFrame(csv_data)
    df_csv.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("✨ 処理完了！")
    print(f"💾 JSON結果を保存: {Path(OUTPUT_JSON).resolve()}")
    print(f"💾 CSV結果を保存: {Path(OUTPUT_CSV).resolve()}\n")


if __name__ == "__main__":
    main()
