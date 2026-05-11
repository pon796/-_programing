"""
春日部駅高架化による短縮効果分析プログラム
GIS を用いて、高架化による徒歩時間短縮を定量的に計算
"""

import osmnx as ox
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import unary_union
import networkx as nx
import warnings
import random
import math

warnings.filterwarnings('ignore')

# ==================== 設定パラメータ ====================
SAMPLE_SIZE = 100  # サンプル数
KASUKABE_STATION_LAT = 35.9749
KASUKABE_STATION_LON = 139.7474
RADIUS_KM = 1.0  # 半径 1km
WALKING_SPEED_KMH = 4.0  # 歩行速度 4km/h
ELEVATED_AREA_BUFFER = 30  # 高架化エリアのバッファ（メートル）

# 通り抜けポイント（緯度、経度）
WAYPOINTS = {
    'P1': (35.9751, 139.7450),  # 5998 Kasukabe
    'P2': (35.9789, 139.7489),  # 4-chōme-2-40
    'P3': (35.9805, 139.7478),  # 4-chōme-1-29
    'P4': (35.9823, 139.7452),  # 3-chōme-2-35
    'P5': (35.9890, 139.7410),  # 6591-1 Kasukabe
    'P6': (35.9654, 139.7519),  # 1-chōme-44-1 Chūō
    'P7': (35.9691, 139.7367),  # 近藤葬儀式場
}

# 高架化エリアの端点
ELEVATED_START = (35.9751, 139.7450)  # 近藤葬儀式場
ELEVATED_END = (35.9691, 139.7367)    # 5998 Kasukabe


def generate_random_points_on_circle(center_lat, center_lon, radius_km, num_points):
    """
    中心座標から半径 radius_km の円周上にランダムに num_points 個の点を生成
    """
    points = []
    earth_radius_km = 6371  # 地球の半径（km）
    
    for _ in range(num_points):
        # ランダムな角度と距離を生成
        angle = random.uniform(0, 2 * math.pi)
        
        # 緯度経度への変換
        lat_offset = (radius_km / earth_radius_km) * (180 / math.pi) * math.cos(angle)
        lon_offset = (radius_km / earth_radius_km) * (180 / math.pi) * math.sin(angle) / math.cos(math.radians(center_lat))
        
        new_lat = center_lat + lat_offset
        new_lon = center_lon + lon_offset
        
        points.append((new_lat, new_lon))
    
    return points


def get_road_network(center_lat, center_lon, radius_m=2000):
    """
    OpenStreetMap から道路ネットワークを取得
    """
    print("📡 OpenStreetMap から道路ネットワークを取得中...")
    
    try:
        # 中心座標から指定距離内の道路ネットワークを取得
        G = ox.graph_from_point(
            (center_lat, center_lon),
            dist=radius_m,
            network_type='drive',
            simplify=True,
            retain_all=False,
            truncate_by_edge=True
        )
        print(f"✅ ネットワーク取得完了: {len(G.nodes())} ノード, {len(G.edges())} エッジ")
        return G
    except Exception as e:
        print(f"❌ エラー: {e}")
        return None


def find_nearest_node(G, lat, lon):
    """
    グラフ G 内で (lat, lon) に最も近いノードを見つける
    """
    try:
        node = ox.distance.nearest_nodes(G, lon, lat)
        return node
    except:
        return None


def calculate_shortest_path_distance(G, start_lat, start_lon, end_lat, end_lon):
    """
    OpenStreetMap の道路ネットワークを使用して最短経路の距離を計算
    """
    try:
        start_node = find_nearest_node(G, start_lat, start_lon)
        end_node = find_nearest_node(G, end_lat, end_lon)
        
        if start_node is None or end_node is None:
            return None
        
        # 最短経路を計算
        try:
            path = nx.shortest_path(G, start_node, end_node, weight='length')
            
            # 経路の総距離を計算
            distance = 0
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                distance += G[u][v][0]['length']
            
            return distance
        except nx.NetworkXNoPath:
            # 経路が見つからない場合
            return None
    except Exception as e:
        print(f"⚠️ 経路計算エラー: {e}")
        return None


def create_elevated_area_polygon(point1, point2, buffer_m=30):
    """
    2点を結ぶ線路沿いのポリゴンを作成（バッファ付き）
    """
    line = LineString([point1, point2])
    # バッファを追加してポリゴン化
    polygon = line.buffer(buffer_m / 111000)  # メートルを度に換算
    return polygon


def analyze_sample(G, elevated_polygon, sample_idx, start_point):
    """
    1つのサンプルを分析
    """
    start_lat, start_lon = start_point
    
    # ステップ 2: 出発点から7つの通り抜けポイントまでの最短経路を計算
    min_distance_to_waypoint = float('inf')
    nearest_waypoint = None
    
    for wp_name, (wp_lat, wp_lon) in WAYPOINTS.items():
        distance = calculate_shortest_path_distance(G, start_lat, start_lon, wp_lat, wp_lon)
        if distance is not None and distance < min_distance_to_waypoint:
            min_distance_to_waypoint = distance
            nearest_waypoint = wp_name
    
    if min_distance_to_waypoint == float('inf'):
        return None
    
    # ステップ 3: 高架化エリアの端点までの最短経路を計算
    # ここでは高架化エリアの始点までの距離を計算
    distance_to_elevated_start = calculate_shortest_path_distance(
        G, start_lat, start_lon, ELEVATED_START[0], ELEVATED_START[1]
    )
    
    if distance_to_elevated_start is None:
        return None
    
    # ステップ 4: 短縮できる距離を計算
    # 簡略版: 高架化エリアを迂回する距離 ≈ 通常経路より短くなる距離
    distance_saved = max(0, min_distance_to_waypoint - distance_to_elevated_start)
    
    # 短縮時間を計算（秒）
    time_saved_seconds = (distance_saved / 1000) / (WALKING_SPEED_KMH) * 3600
    time_saved_minutes = time_saved_seconds / 60
    
    return {
        'sample': sample_idx + 1,
        'start_lat': start_lat,
        'start_lon': start_lon,
        'nearest_waypoint': nearest_waypoint,
        'distance_to_waypoint': min_distance_to_waypoint,
        'distance_to_elevated': distance_to_elevated_start,
        'distance_saved': distance_saved,
        'time_saved_seconds': time_saved_seconds,
        'time_saved_minutes': time_saved_minutes,
    }


def main():
    print("\n" + "="*60)
    print("🚉 春日部駅高架化による短縮効果分析")
    print("="*60 + "\n")
    
    # ステップ1: 道路ネットワークを取得
    G = get_road_network(KASUKABE_STATION_LAT, KASUKABE_STATION_LON, radius_m=2000)
    
    if G is None:
        print("❌ 処理を中止します。")
        return
    
    # 高架化エリアのポリゴンを作成
    print("🗺️  高架化エリアの定義...")
    elevated_polygon = create_elevated_area_polygon(ELEVATED_START, ELEVATED_END, ELEVATED_AREA_BUFFER)
    
    # ランダム出発点を生成
    print(f"\n🎲 {SAMPLE_SIZE} 個のランダム出発点を生成中...")
    random_points = generate_random_points_on_circle(
        KASUKABE_STATION_LAT,
        KASUKABE_STATION_LON,
        RADIUS_KM,
        SAMPLE_SIZE
    )
    print(f"✅ {len(random_points)} 個の出発点を生成完了")
    
    # 各サンプルを分析
    print(f"\n📊 {SAMPLE_SIZE} 個のサンプルを分析中...\n")
    
    results = []
    valid_count = 0
    
    for idx, point in enumerate(random_points):
        # 進捗表示
        if (idx + 1) % 10 == 0:
            print(f"  進捗: {idx + 1}/{SAMPLE_SIZE} ({100*(idx+1)/SAMPLE_SIZE:.1f}%)")
        
        result = analyze_sample(G, elevated_polygon, idx, point)
        
        if result is not None:
            results.append(result)
            valid_count += 1
    
    print(f"\n✅ 分析完了: {valid_count}/{SAMPLE_SIZE} サンプルが有効")
    
    if len(results) == 0:
        print("❌ 有効なデータがありません。")
        return
    
    # 結果をデータフレームに変換
    df = pd.DataFrame(results)
    
    # 統計計算
    mean_time_saved = df['time_saved_minutes'].mean()
    median_time_saved = df['time_saved_minutes'].median()
    std_time_saved = df['time_saved_minutes'].std()
    min_time_saved = df['time_saved_minutes'].min()
    max_time_saved = df['time_saved_minutes'].max()
    
    # 結果出力
    print("\n" + "="*60)
    print("【分析結果】")
    print("="*60)
    print(f"サンプル数: {SAMPLE_SIZE}")
    print(f"有効なデータ: {len(results)} サンプル\n")
    
    print("【短縮徒歩時間の統計】")
    print(f"  📍 平均短縮時間: {mean_time_saved:.2f} 分 ({mean_time_saved*60:.1f} 秒)")
    print(f"  📊 中央値: {median_time_saved:.2f} 分")
    print(f"  📈 標準偏差: {std_time_saved:.2f} 分")
    print(f"  ⬇️  最小値: {min_time_saved:.2f} 分")
    print(f"  ⬆️  最大値: {max_time_saved:.2f} 分")
    print("="*60 + "\n")
    
    # CSV に保存
    output_file = 'kasukabe_analysis_results.csv'
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"💾 結果を保存: {output_file}")
    print("\n✨ 処理完了！")


if __name__ == '__main__':
    main()
