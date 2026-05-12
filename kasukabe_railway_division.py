"""
春日部駅周辺を東武スカイツリー線路で分割した2領域にランダムポイントを生成
各ポイントの住所を逆ジオコーディングで取得するシステム
10回繰り返して結果を出力
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
import json

# ==================== 設定パラメータ ====================
KASUKABE_STATION_LAT = 35.9749
KASUKABE_STATION_LON = 139.7474
RADIUS_KM = 1.0  # 半径 1km

# 線路の端点
RAILWAY_START = (35.9691, 139.7367)    # 近藤葬儀式場
RAILWAY_END = (35.9751, 139.7450)      # 5998 Kasukabe

# OSMnx での検索タグ
RAILWAY_TAGS = {'railway': 'rail'}  # 東武スカイツリー線

# 繰り返し回数
NUM_ITERATIONS = 10


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
            
            # ジオメトリを抽出
            railway_lines = []
            for idx, row in features.iterrows():
                if row.geometry.geom_type in ['LineString', 'MultiLineString']:
                    railway_lines.append(row.geometry)
            
            if railway_lines:
                # 全ての線路を結合
                combined_railway = unary_union(railway_lines)
                return combined_railway
            else:
                # シンプルな線路ラインを作成（2点を結ぶ直線）
                simple_line = LineString([(start_lon, start_lat), (end_lon, end_lat)])
                return simple_line
                
        except Exception as e:
            simple_line = LineString([(start_lon, start_lat), (end_lon, end_lat)])
            return simple_line
            
    except Exception as e:
        # フォールバック: シンプルなライン
        simple_line = LineString([(start_lon, start_lat), (end_lon, end_lat)])
        return simple_line


def split_area_by_railway(search_polygon, railway_line):
    """
    検索エリアを線路で2つの領域に分割
    """
    try:
        # 線路で領域を分割
        railway_buffer = railway_line.buffer(0.0001)  # 小さなバッファ
        
        # 差分演算で分割
        divided_area = split(search_polygon, railway_line)
        
        if len(divided_area.geoms) >= 2:
            area1 = divided_area.geoms[0]
            area2 = divided_area.geoms[1]
            return area1, area2
        else:
            return fallback_split(search_polygon, railway_line)
    except Exception as e:
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


def convert_to_googlemaps_address(lat, lon, raw_address):
    """
    Nominatim のアドレスを Google Maps で検索できる形式に変換
    """
    # 基本的には「座標, 住所」の形式に変換
    # Google Maps は「latitude, longitude」や「address」で検索可能
    
    # 日本語アドレスへの変換を試みる
    try:
        # Nominatim の返す英語アドレスを簡潔にしつつ、座標も含める
        # Google Maps フォーマット: 緯度,経度 または 住所
        
        # 住所から不要な部分を削除
        address_parts = raw_address.split(',')
        
        # 最後の国名（Japan）を除いて、前から3-4個の要素を取得
        filtered_parts = [part.strip() for part in address_parts if part.strip() and 'Japan' not in part]
        
        if len(filtered_parts) > 0:
            # 最後の2-3要素を使用（市区町村、通り名など）
            important_parts = filtered_parts[-3:] if len(filtered_parts) >= 3 else filtered_parts
            google_maps_address = ', '.join(important_parts)
        else:
            google_maps_address = raw_address
        
        # フォーマット: 「住所 (座標)」
        formatted_address = f"{google_maps_address} ({lat:.6f}, {lon:.6f})"
        
        return formatted_address
    except Exception as e:
        # エラー時は座標をベースにしたアドレスを返す
        return f"{lat:.6f}, {lon:.6f}"


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
                time.sleep(0.5)  # 0.5秒待機
            else:
                return f"{lat:.6f}, {lon:.6f}"
    
    return f"{lat:.6f}, {lon:.6f}"


def main():
    print("\n" + "="*80)
    print("🚉 春日部駅周辺を線路で分割し、2つの領域にランダムポイントを生成")
    print("="*80 + "\n")
    
    # 初期化（1回目のみ）
    print("📍 初期化: 検索エリア、線路を取得中...\n")
    
    search_polygon, bounds = create_search_area_polygon(
        KASUKABE_STATION_LAT,
        KASUKABE_STATION_LON,
        RADIUS_KM
    )
    print(f"✅ 検索エリア作成完了: 半径 {RADIUS_KM} km")
    
    railway_line = get_railway_line(
        RAILWAY_START[0], RAILWAY_START[1],
        RAILWAY_END[0], RAILWAY_END[1]
    )
    print(f"✅ 東武スカイツリー線路を取得完了\n")
    
    # 10回繰り返し実行
    all_results = []
    
    for iteration in range(NUM_ITERATIONS):
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"【実行 {iteration + 1}/{NUM_ITERATIONS}】")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        
        # エリアを分割
        area1, area2 = split_area_by_railway(search_polygon, railway_line)
        
        # 各領域にランダムポイントを生成
        point1 = generate_random_point_in_polygon(area1)
        lat1, lon1 = point1.y, point1.x
        
        point2 = generate_random_point_in_polygon(area2)
        lat2, lon2 = point2.y, point2.x
        
        # 住所を取得
        print(f"   📍 領域1 の住所を取得中... ", end='', flush=True)
        raw_address1 = get_address_from_coordinates(lat1, lon1)
        address1 = convert_to_googlemaps_address(lat1, lon1, raw_address1)
        print(f"✅\n   {address1}\n")
        
        print(f"   📍 領域2 の住所を取得中... ", end='', flush=True)
        raw_address2 = get_address_from_coordinates(lat2, lon2)
        address2 = convert_to_googlemaps_address(lat2, lon2, raw_address2)
        print(f"✅\n   {address2}\n")
        
        # 結果を保存
        result = {
            'iteration': iteration + 1,
            'area1': {
                'latitude': lat1,
                'longitude': lon1,
                'raw_address': raw_address1,
                'google_maps_address': address1
            },
            'area2': {
                'latitude': lat2,
                'longitude': lon2,
                'raw_address': raw_address2,
                'google_maps_address': address2
            }
        }
        all_results.append(result)
        
        # Nominatim のレート制限を回避するため、少し待機
        if iteration < NUM_ITERATIONS - 1:
            time.sleep(1)
    
    # 最終結果を表示
    print("\n" + "="*80)
    print("【 最終結果 - 10組の住所ペア 】")
    print("="*80 + "\n")
    
    for result in all_results:
        iteration = result['iteration']
        area1_addr = result['area1']['google_maps_address']
        area2_addr = result['area2']['google_maps_address']
        
        print(f"【{iteration}】")
        print(f"  領域1: {area1_addr}")
        print(f"  領域2: {area2_addr}")
        print()
    
    print("="*80)
    print("✨ 処理完了！\n")
    
    # 結果を JSON ファイルに保存
    with open('kasukabe_random_points_results_10.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print("💾 詳細結果を保存: kasukabe_random_points_results_10.json\n")
    
    # 簡潔な CSV にも保存
    csv_data = []
    for result in all_results:
        csv_data.append({
            '実行番号': result['iteration'],
            '領域1_住所': result['area1']['google_maps_address'],
            '領域2_住所': result['area2']['google_maps_address'],
            '領域1_緯度': result['area1']['latitude'],
            '領域1_経度': result['area1']['longitude'],
            '領域2_緯度': result['area2']['latitude'],
            '領域2_経度': result['area2']['longitude'],
        })
    
    df_csv = pd.DataFrame(csv_data)
    df_csv.to_csv('kasukabe_random_points_results_10.csv', index=False, encoding='utf-8-sig')
    
    print("💾 CSV 結果を保存: kasukabe_random_points_results_10.csv\n")


if __name__ == '__main__':
    main()
