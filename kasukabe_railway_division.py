"""
春日部駅周辺を東武スカイツリー線路で分割した2領域にランダムポイントを生成
各ポイントの住所を逆ジオコーディングで取得するシステム
"""

import osmnx as ox
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, LineString, Polygon, box
from shapely.ops import split, unary_union
import random
import math
import time
from geopy.geocoders import Nominatim

# ==================== 設定パラメータ ====================
KASUKABE_STATION_LAT = 35.9749
KASUKABE_STATION_LON = 139.7474
RADIUS_KM = 1.0  # 半径 1km

# 線路の端点
RAILWAY_START = (35.9691, 139.7367)    # 近藤葬儀式場
RAILWAY_END = (35.9751, 139.7450)      # 5998 Kasukabe

# OSMnx での検索タグ
RAILWAY_TAGS = {'railway': 'rail'}  # 東武スカイツリー線


def create_search_area_polygon(center_lat, center_lon, radius_km):
    """
    中心座標から半径 radius_km の円形ポリゴンを作成
    """
    # 緯度経度を度から度への大ざっぱな変換（正確性は低いが、この目的では十分）
    earth_radius_km = 6371
    lat_offset = (radius_km / earth_radius_km) * (180 / math.pi)
    lon_offset = (radius_km / earth_radius_km) * (180 / math.pi) / math.cos(math.radians(center_lat))
    
    # バウンディングボックスを作成
    min_lat = center_lat - lat_offset
    max_lat = center_lat + lat_offset
    min_lon = center_lon - lon_offset
    max_lon = center_lon + lon_offset
    
    # 円ポリゴンを作成（北から南への楕円形で近似）
    circle_points = []
    for angle in np.linspace(0, 2 * np.pi, 360):
        x = center_lon + lon_offset * 0.9 * np.cos(angle)
        y = center_lat + lat_offset * 0.9 * np.sin(angle)
        circle_points.append((x, y))
    
    circle_polygon = Polygon(circle_points)
    return circle_polygon, (min_lat, max_lat, min_lon, max_lon)


def get_railway_line(start_lat, start_lon, end_lat, end_lon, search_distance=2000):
    """
    OpenStreetMap から東武スカイツリー線路を取得
    線路をポリラインとして抽出
    """
    print("🚄 東武スカイツリー線路を検索中...\n")
    
    try:
        # 線路のジオメトリを取得
        # 複数のソースから線路を検索
        ox.settings.log_console = False
        
        # 検索エリアの中心
        center_lat = (start_lat + end_lat) / 2
        center_lon = (start_lon + end_lon) / 2
        
        # railway タグを持つ全ての線路を取得
        tags = {'railway': True}
        try:
            features = ox.features_from_point(
                (center_lat, center_lon),
                tags,
                dist=search_distance
            )
            print(f"✅ 線路データを取得: {len(features)} 件")
            
            # ジオメトリを抽出
            railway_lines = []
            for idx, row in features.iterrows():
                if row.geometry.geom_type in ['LineString', 'MultiLineString']:
                    railway_lines.append(row.geometry)
            
            if railway_lines:
                # 全ての線路を結合
                combined_railway = unary_union(railway_lines)
                print(f"✅ 線路を結合完了")
                return combined_railway
            else:
                print("⚠️ 線路ジオメトリが見つかりません。代わりにシンプルなラインを使用します。")
                # シンプルな線路ラインを作成（2点を結ぶ直線）
                simple_line = LineString([(start_lon, start_lat), (end_lon, end_lat)])
                return simple_line
                
        except Exception as e:
            print(f"⚠️ エラー: {e}")
            print("   シンプルなラインを使用します。")
            simple_line = LineString([(start_lon, start_lat), (end_lon, end_lat)])
            return simple_line
            
    except Exception as e:
        print(f"❌ エラー: {e}")
        # フォールバック: シンプルなライン
        simple_line = LineString([(start_lon, start_lat), (end_lon, end_lat)])
        return simple_line


def split_area_by_railway(search_polygon, railway_line):
    """
    検索エリアを線路で2つの領域に分割
    """
    print("\n🗺️  検索エリアを線路で分割中...\n")
    
    try:
        # 線路で領域を分割
        # 線路の周辺にバッファを作成して、線路が交差するようにする
        railway_buffer = railway_line.buffer(0.0001)  # 小さなバッファ
        
        # 差分演算で分割
        divided_area = split(search_polygon, railway_line)
        
        if len(divided_area.geoms) >= 2:
            area1 = divided_area.geoms[0]
            area2 = divided_area.geoms[1]
            print(f"✅ 領域を2つに分割完了")
            print(f"   領域1 面積: {area1.area:.6f}")
            print(f"   領域2 面積: {area2.area:.6f}")
            return area1, area2
        else:
            print("⚠️ 完全な分割ができませんでした。代替方法を使用します。")
            # 線路を使用した代替分割
            return fallback_split(search_polygon, railway_line)
    except Exception as e:
        print(f"⚠️ 分割エラー: {e}")
        return fallback_split(search_polygon, railway_line)


def fallback_split(search_polygon, railway_line):
    """
    分割がうまくいかない場合の代替方法
    線路の周辺でバッファを作成して、ポリゴンを分割
    """
    try:
        # 線路の両側にバッファを作成
        buffer_distance = 0.002  # 度単位（約200m）
        
        # 線路からのバッファ領域を作成
        railway_buffer_left = railway_line.buffer(buffer_distance)
        railway_buffer_right = railway_line.buffer(-buffer_distance)
        
        # 領域1: 線路の左側
        area1 = search_polygon.difference(railway_buffer_left)
        # 領域2: 線路の右側
        area2 = search_polygon.difference(railway_buffer_right)
        
        # 領域が小さすぎる場合の補正
        if area1.area < 0.00001 or area2.area < 0.00001:
            # 線路を中心に左右に分割
            coords = list(railway_line.coords)
            if len(coords) >= 2:
                # 線路に垂直な分割線を作成
                lat_range = coords[-1][1] - coords[0][1]
                lon_range = coords[-1][0] - coords[0][0]
                
                # 中点を通る垂直線
                mid_lat = (coords[0][1] + coords[-1][1]) / 2
                mid_lon = (coords[0][0] + coords[-1][0]) / 2
                
                # 垂直線を作成（北南方向）
                perpendicular = LineString([
                    (mid_lon - 0.05, mid_lat),
                    (mid_lon + 0.05, mid_lat)
                ])
                
                divided = split(search_polygon, perpendicular)
                if len(divided.geoms) >= 2:
                    return divided.geoms[0], divided.geoms[1]
        
        return area1, area2
    except Exception as e:
        print(f"❌ 代替分割もエラー: {e}")
        # 最終手段: 中心で単純に分割
        bounds = search_polygon.bounds
        mid_lat = (bounds[1] + bounds[3]) / 2
        split_line = LineString([(bounds[0], mid_lat), (bounds[2], mid_lat)])
        divided = split(search_polygon, split_line)
        return divided.geoms[0], divided.geoms[1]


def generate_random_point_in_polygon(polygon, max_attempts=100):
    """
    ポリゴン内にランダムなポイントを生成
    """
    bounds = polygon.bounds
    
    for attempt in range(max_attempts):
        x = random.uniform(bounds[0], bounds[2])
        y = random.uniform(bounds[1], bounds[3])
        point = Point(x, y)
        
        if polygon.contains(point):
            return point
    
    # max_attempts 回失敗した場合、ポリゴンの代表点を返す
    return polygon.representative_point()


def get_address_from_coordinates(lat, lon, max_retries=3):
    """
    緯度経度から住所を取得（逆ジオコーディング）
    """
    geolocator = Nominatim(user_agent="kasukabe_random_points")
    
    for attempt in range(max_retries):
        try:
            location = geolocator.reverse(f"{lat}, {lon}", language='en')
            address = location.address
            return address
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"   ⏳ リトライ {attempt + 1}/{max_retries - 1}...")
                time.sleep(1)  # 1秒待機
            else:
                return f"住所取得失敗: {lat:.6f}, {lon:.6f}"
    
    return f"住所取得失敗: {lat:.6f}, {lon:.6f}"


def main():
    print("\n" + "="*70)
    print("🚉 春日部駅周辺を線路で分割し、2つの領域にランダムポイントを生成")
    print("="*70 + "\n")
    
    # ステップ 1: 検索エリアを作成
    print("📍 ステップ 1: 検索エリア（半径1km の円）を作成中...\n")
    search_polygon, bounds = create_search_area_polygon(
        KASUKABE_STATION_LAT,
        KASUKABE_STATION_LON,
        RADIUS_KM
    )
    print(f"✅ 検索エリア作成完了")
    print(f"   中心: ({KASUKABE_STATION_LAT}, {KASUKABE_STATION_LON})")
    print(f"   半径: {RADIUS_KM} km\n")
    
    # ステップ 2: 線路を取得
    print("📍 ステップ 2: 東武スカイツリー線路を取得中...\n")
    railway_line = get_railway_line(
        RAILWAY_START[0], RAILWAY_START[1],
        RAILWAY_END[0], RAILWAY_END[1]
    )
    print()
    
    # ステップ 3: エリアを分割
    print("📍 ステップ 3: エリアを線路で分割中...\n")
    area1, area2 = split_area_by_railway(search_polygon, railway_line)
    
    # ステップ 4: 各領域にランダムポイントを生成
    print("\n📍 ステップ 4: 各領域にランダムポイントを生成中...\n")
    
    print("   領域1 にランダムポイントを生成...")
    point1 = generate_random_point_in_polygon(area1)
    lat1, lon1 = point1.y, point1.x
    print(f"   ✅ 座標: ({lat1:.6f}, {lon1:.6f})")
    
    print("   領域2 にランダムポイントを生成...")
    point2 = generate_random_point_in_polygon(area2)
    lat2, lon2 = point2.y, point2.x
    print(f"   ✅ 座標: ({lat2:.6f}, {lon2:.6f})\n")
    
    # ステップ 5: 住所を取得
    print("📍 ステップ 5: 各ポイントの住所を逆ジオコーディングで取得中...\n")
    
    print("   領域1 の住所を取得中...")
    address1 = get_address_from_coordinates(lat1, lon1)
    print(f"   ✅ 住所: {address1}\n")
    
    print("   領域2 の住所を取得中...")
    address2 = get_address_from_coordinates(lat2, lon2)
    print(f"   ✅ 住所: {address2}\n")
    
    # 結果を出力
    print("="*70)
    print("【 最終結果 】")
    print("="*70)
    print(f"\n【領域1】")
    print(f"  座標: ({lat1:.6f}, {lon1:.6f})")
    print(f"  🏠 住所: {address1}")
    print(f"\n【領域2】")
    print(f"  座標: ({lat2:.6f}, {lon2:.6f})")
    print(f"  🏠 住所: {address2}")
    print("\n" + "="*70)
    print("✨ 処理完了！\n")
    
    # 結果を JSON ファイルに保存
    results = {
        'area1': {
            'latitude': lat1,
            'longitude': lon1,
            'address': address1
        },
        'area2': {
            'latitude': lat2,
            'longitude': lon2,
            'address': address2
        }
    }
    
    import json
    with open('kasukabe_random_points_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("💾 結果を保存: kasukabe_random_points_results.json\n")
    
    return results


if __name__ == '__main__':
    main()
