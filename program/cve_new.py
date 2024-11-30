"""
マスターベクトルを用いて色素成分分離を行い画像を出力する
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import warnings
warnings.filterwarnings('ignore')
import cv2
import glob
import numpy as np
from pathlib import Path
import rawpy
from PIL import Image
from PIL.ExifTags import TAGS
import mediapipe as mp  
from tqdm import tqdm  
import matplotlib.pyplot as plt
import plotly.graph_objects as go
# MediapipeのFaceDetectionを初期化
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import numpy as np

#========================================================#
# 可視化用関数#
#=======================================================
def plot_histogram(data, title="Histogram", xlabel="Value", ylabel="Frequency", bins=30, color="blue", alpha=0.7):
    """
    データのヒストグラムをプロットする。

    Parameters:
        data (numpy.array): ヒストグラムに使用するデータ（1次元配列）。
        title (str): グラフのタイトル。
        xlabel (str): X軸のラベル。
        ylabel (str): Y軸のラベル。
        bins (int): ヒストグラムの棒の数。
        color (str): 棒の色。
        alpha (float): 棒の透明度。
    """
    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=bins, range=(-2, 3), color=color, alpha=alpha)  # 範囲を -2 から 3 に指定
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
    
    
def plot_3d__skinFlat(skin_flat,melanin_vector, hemoglobin_vector,vec):
    """
    3次元プロットを作成してSのRGB成分を可視化する関数。
    
    入力:
        S: (3, height, width) のnumpy配列
    """
    # SのRGB成分を平坦化して取得
    x = skin_flat[0].flatten()  # R成分
    y = skin_flat[1].flatten()  # G成分
    z = skin_flat[2].flatten()  # B成分
    
    
    # 3Dプロットの設定
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x.flatten(),y.flatten(),z.flatten(), 
               c='#FFDAB9', marker='o', alpha=0.5, label="Projected Points on Plane",s=1)
    
    melanin_vector = np.array(melanin_vector)
    hemoglobin_vector = np.array(hemoglobin_vector)

    # メッシュグリッドのパラメータ範囲を定義
    u = np.linspace(0, 2, 10)
    v = np.linspace(0, 2, 10)
    U, V = np.meshgrid(u, v)
    plane_x =  U * hemoglobin[0] + V * melanin[0]
    plane_y = U * hemoglobin[1] + V * melanin[1]
    plane_z =  U * hemoglobin[2] + V * melanin[2]
    ax.plot_surface(plane_x, plane_y, plane_z, color='lightblue', alpha=0.5)
    
    ax.quiver(0, 0, 0, melanin_vector[0], melanin_vector[1], melanin_vector[2], 
              color='brown', label='Melanin Vector', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, hemoglobin_vector[0], hemoglobin_vector[1], hemoglobin_vector[2], 
              color='yellow', label='Hemoglobin Vector', arrow_length_ratio=0.1)
    
    ax.set_xlabel('-logR Component')
    ax.set_ylabel('-logG Component')
    ax.set_zlabel('-logB Component')
    ax.set_title("3D Plot of"+" skinFlat")

    plt.show()

def plot_3d_skinFlat_plotly(skin_flat, melanin_vector, hemoglobin_vector, vec, max_points=5000):
    """
    3次元プロットをPlotlyでインタラクティブに表示

    Parameters:
        skin_flat: (3, height, width) のnumpy配列。
        melanin_vector: メラニンベクトル。
        hemoglobin_vector: ヘモグロビンベクトル。
        vec: 色成分ベクトル。
        max_points: 表示する最大点数。
    """
    # SのRGB成分を平坦化して取得
    x = skin_flat[0].flatten()
    y = skin_flat[1].flatten()
    z = skin_flat[2].flatten()
    
    # サンプリング
    if len(x) > max_points:
        indices = np.random.choice(len(x), max_points, replace=False)
        x, y, z = x[indices], y[indices], z[indices]

    # 点群を追加
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(size=2, color='peachpuff', opacity=0.5)
    )])

    # 平面を追加
    melanin_vector = np.array(melanin_vector)
    hemoglobin_vector = np.array(hemoglobin_vector)
    u = np.linspace(0, 6, 30)
    v = np.linspace(0, 6, 30)
    U, V = np.meshgrid(u, v)
    plane_x = U * hemoglobin_vector[0] + V * melanin_vector[0]
    plane_y = U * hemoglobin_vector[1] + V * melanin_vector[1]
    plane_z = U * hemoglobin_vector[2] + V * melanin_vector[2]
    fig.add_trace(go.Surface(
        x=plane_x, y=plane_y, z=plane_z,
        colorscale='Blues', opacity=0.5
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title='-logR Component',
            yaxis_title='-logG Component',
            zaxis_title='-logB Component'
        ),
        title="3D Plot of SkinFlat (Plotly)"
    )
    fig.show()

# ================================================================#
# 顔マスク用関数　#
# ================================================================#
def create_black_mask(image_rgb):
    """
    黒色部分以外を1、黒色部分を0とするマスクを生成する関数
    """
    # 各ピクセルが完全な黒かどうかを判定（全てのチャネルが0）
    mask = np.all(image_rgb == [0, 0, 0], axis=-1)
    # 黒色以外の部分を1に、黒色部分を0にする
    inverse_mask = np.logical_not(mask).astype(np.float32)
    return inverse_mask


def create_face_mask(image_rgb,erosion_size=5):
    """
    Mediapipeを使って顔と耳の領域を含むマスクを作成する関数
    """
    image_height, image_width = image_rgb.shape[:2]
    face_mask = np.zeros((image_height, image_width), dtype=np.float32)  # マスク初期化

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 顔のランドマークを2D座標に変換し、輪郭を描画
                points = [(int(landmark.x * image_width), int(landmark.y * image_height))
                          for landmark in face_landmarks.landmark]

                # 輪郭をポリゴンで描画
                convex_hull = cv2.convexHull(np.array(points))
                cv2.fillConvexPoly(face_mask, convex_hull, 1)
                
        # 収縮処理でマスクを輪郭より内側に調整
    kernel = np.ones((erosion_size, erosion_size), np.uint8)
    inner_face_mask = cv2.erode(face_mask, kernel, iterations=1)

    return inner_face_mask

def create_hsv_mask(image_rgb, hue_range=(0, 180), saturation_range=(0, 255), value_range=(0, 255)):
    """
    HSV色空間で特定の範囲に基づいたマスクを作成する関数。

    Parameters:
        image_rgb (numpy.ndarray): RGB画像。
        hue_range (tuple): 色相(H)の範囲 (0-180)。
        saturation_range (tuple): 彩度(S)の範囲 (0-255)。
        value_range (tuple): 明度(V)の範囲 (0-255)。

    Returns:
        mask (numpy.ndarray): HSVマスク（0または1の値を持つ2D配列）。
    """
    # RGB画像をBGRに変換
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # BGR画像をHSV色空間に変換
    hsv_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # 範囲指定でマスクを作成
    lower_bound = np.array([hue_range[0], saturation_range[0], value_range[0]])
    upper_bound = np.array([hue_range[1], saturation_range[1], value_range[1]])
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

    # マスクを二値化して0または1にする
    mask = (mask > 0).astype(np.float32)
    return mask


# ================================================================#
# 画像読み込み　出力用関数　#
# ================================================================#
def save_image(path, image):
    """"
    画像を保存するための関数
    # """
    image_out = np.clip(image, 0, 1)

    # image_out=image
    image_out = cv2.cvtColor((image_out * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image_out)
    return

def get_image_rotation_info(image_path):
    """"
    iphoneで撮影した画像の回転情報を読み込む
    意図しない回転に対応するため
    入力:image_path 画像のパス
    出力：orientation 回転情報
    """
    img = Image.open(image_path)
    # EXIFデータを取得
    exif_data = img.getexif()

    if not exif_data:
        return None
    exif = {TAGS.get(tag): value for tag, value in exif_data.items()}

    # Orientationタグを取得（回転情報が含まれている）
    orientation = exif.get('Orientation', None)
    return orientation

def read_cr2_image(file_path):
    """
    .CR2ファイルを読み込み、RGB画像に変換する
    """
    # CR2を読み込み
    with rawpy.imread(file_path) as raw:
        rgb_image = raw.postprocess(
                use_camera_wb=True,  # カメラのホワイトバランスを使用
                half_size=False,      # 解像度を落とさない
                no_auto_bright=True,  # 自動輝度調整を無効化
            )  # デモザイク処理してRGB画像に変換

    return rgb_image

#==============================================================
# バイアス調整用関数
# =============================================================

def find_optimal_offset(skin_flat, melanin, hemoglobin):
    """
    点群をメラニンとヘモグロビンベクトルの線形結合で表される範囲に多く含まれるよう移動する最適なオフセットを計算。

    Parameters:
        skin_flat: (3, height, width) のnumpy配列。点群。
        melanin: メラニンベクトル (3要素)。
        hemoglobin: ヘモグロビンベクトル (3要素)。

    Returns:
        optimal_offset: 点群を移動させる最適なベクトル (3要素)。
    """
    # 点群を3次元座標のリストに変換
    skin_flat_points = skin_flat.reshape(3, -1).T  # (N, 3)
    
    # メラニン・ヘモグロビンベクトルの基底を定義
    basis = np.array([melanin, hemoglobin]).T  # (3, 2)

    # 点が線形結合の範囲内に入るかをチェックする関数
    def is_within_range(points):
        """
        点群がメラニン・ヘモグロビンの正の線形結合で表されるか判定。
        """
        coefficients, residuals, rank, s = np.linalg.lstsq(basis, points.T, rcond=None)
        alpha, beta = coefficients
        return (alpha > 0) & (beta > 0)

    # 目的関数: 範囲内に入る点の割合を最大化
    def objective(offset):
        moved_points = skin_flat_points + offset  # 点群を移動
        within_range = is_within_range(moved_points)
        return -np.sum(within_range) + np.linalg.norm(offset)# 範囲内に入る点の数を最大化

    # 初期値（オフセットの初期推定）
    O0 = np.zeros(3)

    # 最適化実行
    result = minimize(objective, O0, method='Nelder-Mead')
    
    if not result.success:
        raise ValueError("Optimization failed.")
    
    # 最適な移動ベクトル
    optimal_offset = result.x
    return optimal_offset

#==============================================================#
# 色素成分分離メイン関数
# =============================================================
def makeSkinSeparation(INPUT_DIR,input_image_list,OUTPUT_DIR,vector,mask_type= 'face'):
    """"
    入力画像をメラニン・ヘモグロビン・陰影画像に分離する関数
    入力：input_image_list 画像のリスト
        OUTPUT_DIR: 出力先フォルダへのパス
        vector :メラニン・ヘモグロビンベクトル
    出力　色素成分分離後画像
    """
    
    melanin =vector[0]
    hemoglobin=vector[1]
    shading=vector[2]
    
    print('\n==== Start ====')
    print(input_image_list)
    for input_image_path in tqdm(input_image_list,desc="Processing Images",unit="image"):

        print(input_image_path)
        image_basename = os.path.splitext(os.path.basename(input_image_path))[0]
        image_type = os.path.splitext(os.path.basename(input_image_path))[1]

        #===============================================================================
        # 画像の読み込み
        #===============================================================================
        try:
            if image_type == '.npy': # Linear RGB, HDR after AIRAW and ReAE.
                image_rgb = 0.8 * np.load(input_image_path)
                image_rgb = 255.0 * image_rgb
                image_rgb = cv2.resize(image_rgb, None, fx=1/2, fy=1/2, interpolation=cv2.INTER_CUBIC)
                image_rgb = image_rgb.clip(min=0.0)
            
            elif image_type ==".CR2":
                print("CR2")
                image_rgb = read_cr2_image(input_image_path)
                image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)  # RGB -> BGRに変換
                OUTPUT_PNG_DIR = INPUT_DIR+"\\png\\"
                os.makedirs(OUTPUT_PNG_DIR, exist_ok=True)
                cv2.imwrite(OUTPUT_PNG_DIR+"\\"+image_basename+".png",image_bgr)
            else:
                image_rgb = cv2.cvtColor(cv2.imread(input_image_path, -1), cv2.COLOR_BGR2RGB)
                rotation_info = get_image_rotation_info(input_image_path)
                if rotation_info ==6:
                    image_rgb = cv2.rotate(image_rgb,cv2.ROTATE_90_CLOCKWISE)

                
        except:
            Exception('ERROR: Input image file was not found.')

        image_height = image_rgb.shape[0]
        image_width = image_rgb.shape[1]
        

        
        #===============================================================================
        # 画像調整用パラメーターの設定
        #===============================================================================
        # γ 補正用パラメータの取得
        ##固定値
        
        aa = 1
        bb = 0
        gamma =1 
        cc = 0
        gg = [1, 1, 1]
        DC = 1 / 255
        
        # 色ベクトルと照明強度ベクトル
        vec = np.zeros((3, 3), dtype=np.float32)
        vec[0] = np.array([1.0, 1.0, 1.0])
        vec[1] = melanin
        vec[2] = hemoglobin

        #===============================================================================
        # 画像情報を濃度空間へ
        #===============================================================================

        # 配列の初期化
        linearSkin = np.zeros_like(image_rgb, dtype=np.float32).transpose(2,0,1)
        S = np.zeros_like(image_rgb, dtype=np.float32).transpose(2,0,1)

        # 画像の補正 (画像の最大値を1に正規化)
        
        skin = image_rgb.transpose(2,0,1).astype(np.float32)
        for i in range(3):
            linearSkin[i] = np.power(((skin[i]-cc)/aa), (1/gamma)-bb)/gg[i]/255

        # 顔の領域を含むマスクを作成
        if mask_type == "face":
            print("Using face mask.")
            img_mask = create_face_mask(image_rgb)
        elif mask_type == "black":
            print("Using black mask.")
            img_mask = create_black_mask(image_rgb)
        elif mask_type == "hsv":
            print("Using HSV mask.")
            # 色相、彩度、明度の範囲を指定
            hue_range = (0, 50)  
            saturation_range = (50, 255)
            value_range = (50, 255)
            img_mask = create_hsv_mask(image_rgb, hue_range, saturation_range, value_range)
        elif mask_type == "face-hsv":
            print("Using HSV mask.")
            # 色相、彩度、明度の範囲を指定
            hue_range = (0, 50)  
            saturation_range = (50, 255)
            value_range = (50, 255)
            img_mask1 = create_hsv_mask(image_rgb, hue_range, saturation_range, value_range)
            img_mask2 = create_face_mask(image_rgb)
            
            img_mask = img_mask1 * img_mask2
        elif mask_type == "black-hsv":
            print("black HSV mask.")
            # 色相、彩度、明度の範囲を指定
            hue_range = (0, 50)  
            saturation_range = (50, 255)
            value_range = (50, 255)
            img_mask1 = create_hsv_mask(image_rgb, hue_range, saturation_range, value_range)
            img_mask2 = create_black_mask(image_rgb)
            
            img_mask = img_mask1 * img_mask2
        else:
            raise ValueError(f"Invalid mask_type: {mask_type}. Choose 'face' or 'black'.")
        
        img_mask = np.repeat(img_mask[np.newaxis, :, :], 3, axis=0)  # 3チャンネルに拡張
        img_mask2 = (1 / 255) + np.zeros_like(img_mask, dtype=np.float32)

        linearSkin = np.zeros_like(image_rgb, dtype=np.float32).transpose(2, 0, 1)
        S = np.zeros_like(image_rgb, dtype=np.float32).transpose(2, 0, 1)

        skin = image_rgb.transpose(2, 0, 1).astype(np.float32)
        for i in range(3):
            linearSkin[i] = np.power((skin[i] / 2), 1) / 255


        # 濃度空間 (log空間) へ
        S = -np.log(linearSkin + img_mask2) * img_mask

        
        #===============================================================================
        # 陰影成分の除去
        #===============================================================================
        
        # 肌色分布平面の法線 = 2つの色ベクトルの外積
        # 平面から法線を求める式 (vec(1,:) = [1 1 1] なので式には考慮せず)
        norm = [
            vec[1,1]*vec[2,2]-vec[1,2]*vec[2,1],
            vec[1,2]*vec[2,0]-vec[1,0]*vec[2,2],
            vec[1,0]*vec[2,1]-vec[1,1]*vec[2,0],
            ]

        # 照明ムラ方向と平行な成分をとおる直線と肌色分布平面との交点を求める
        # housen：肌色分布平面の法線
        # S：濃度空間でのRGB
        # vec：独立成分ベクトル
        t = -(norm[0]*S[0] + norm[1]*S[1] + norm[2]*S[2]) / (norm[0]*vec[0,0]+norm[1]*vec[0,1]+norm[2]*vec[0,2])

        # 陰影除去
        # skin_flat：陰影除去したメラヘモベクトルの平面
        # rest：陰影成分
        skin_flat = (t[np.newaxis,:,:].transpose(1,2,0)*vec[0]).transpose(2,0,1) + S
        rest = S - skin_flat
        
        # # 生成されたマスクの保存
        # mask_output_path = os.path.join(OUTPUT_DIR, f"{image_basename}_mask.png")
        # cv2.imwrite(mask_output_path, (img_mask.transpose(1, 2, 0) * 255).astype(np.uint8))
        # print(f"Mask saved to: {mask_output_path}")

        # # マスク適用後の画像の保存
        # masked_image = (image_rgb * img_mask.transpose(1, 2, 0)).astype(np.uint8)
        # masked_image_output_path = os.path.join(OUTPUT_DIR, f"{image_basename}_masked.png")
        # cv2.imwrite(masked_image_output_path, cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))
        # print(f"Masked image saved to: {masked_image_output_path}")
        
        
        plot_3d_skinFlat_plotly(skin_flat,melanin,hemoglobin,vec)
        optimal_offset = find_optimal_offset(skin_flat,melanin,hemoglobin)
        MinSkin =optimal_offset
        
        print(optimal_offset)
        for i in range(3):
            skin_flat[i] = skin_flat[i] + MinSkin[i]
            
        plot_3d_skinFlat_plotly(skin_flat,melanin,hemoglobin,vec)
        #===============================================================================
        # 色素濃度の計算
        #===============================================================================
        # 混合行列と分離行列
        CompExtM = np.linalg.pinv(np.vstack([melanin, hemoglobin]).transpose())

        # 各画素の色素濃度の取得
        #　　濃度分布 ＝ [メラニン色素； ヘモグロビン色素]
        #　　　　　　 ＝ 肌色ベクトル (陰影除去後) × 分離行列
        Compornent = np.dot(CompExtM, skin_flat.reshape(3, image_height * image_width))
        Compornent = Compornent.reshape(2, image_height, image_width)
        Comp = np.vstack([Compornent, (rest[0])[np.newaxis,:,:]])

        #===============================================================================
        # 色素成分分離画像の出力
        #===============================================================================
        # 0：メラニン成分 1：ヘモグロビン成分 2：陰影成分
        L_Mel, L_Hem, L_Sha = Comp
        L_Obj = np.zeros_like(Comp, dtype=np.float32)
        
        for param_index in range(3):

            # 各チャネル情報を取得する。
            if param_index == 0:
                for chn_index in range(3):
                    L_Obj[chn_index] = melanin[chn_index] * L_Mel
                    output_path = '{}_2_Mel.png'.format(image_basename)
            elif param_index == 1:
                for chn_index in range(3):
                    L_Obj[chn_index] = hemoglobin[chn_index] * L_Hem
                    output_path = '{}_3_Hem.png'.format(image_basename)
                
            else:
                for chn_index in range(3):
                    L_Obj[chn_index] =shading[chn_index] * L_Sha
                    output_path = '{}_1_Sha.png'.format(image_basename)

            # 可視化して保存する。最大値と最小値で規格化する
            img = L_Obj.transpose(1,2,0)
            
            img_exp = np.exp(-img) * img_mask.transpose(1,2,0)
            
            # マスク外を灰色に設定
            gray_value = 192 / 255.0 
            img_exp[img_mask.transpose(1, 2, 0) == 0] = gray_value
            
            ef_img =img_exp
            save_image(os.path.join(OUTPUT_DIR, output_path), ef_img)



if __name__ == '__main__':
    
    current= Path(__file__).resolve()
    DATA_DIR =str(current.parents[1] / "data")
    target_list =['check-patch-watanabe']
    
    ## 旧マスターベクトル
    # melanin= [0.087196, 0.471511, 0.877539]
    # hemoglobin = [0.132687, 0.640552, 0.756364]
    ## 新マスターベクトル
    melanin    =[0.2203, 0.4788, 0.8499]
    hemoglobin =[0.4350, 0.6929, 0.5750]
    shading    =[ 1.0000, 1.0000, 1.0000 ]

    vector = [melanin,hemoglobin,shading]

    # マスクタイプを選択 ("face" または "black","hsv",face-hsv)
    mask_type = "black-hsv"  # ここで選択を変更
    for target in target_list:
        OUTPUT_DIR = str(current.parents[1])+"/data/"+target+"/result-cvenew-nooffset/"
        INPUT_DIR  =DATA_DIR+"\\"+target +"\\"
        input_image_list = glob.glob(str(DATA_DIR+"/"+target +"/**.png"))
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        makeSkinSeparation(INPUT_DIR,input_image_list,OUTPUT_DIR,vector,mask_type)


    print('Done.')