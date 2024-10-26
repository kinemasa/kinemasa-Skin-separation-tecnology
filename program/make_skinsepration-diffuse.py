"""
マスターベクトルを用いて色素成分分離を行い画像を出力する
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import warnings
warnings.filterwarnings('ignore')
import cv2
import glob
import numpy as np
from scipy import io, interpolate
from pathlib import Path
import rawpy
from PIL import Image
from PIL.ExifTags import TAGS
import mediapipe as mp  
from tqdm import tqdm  
from natsort import natsorted
from PIL import Image
# Pillowの安全制限を無効化
Image.MAX_IMAGE_PIXELS = None

def split_image(image_path, output_folder, tile_size):
    # 画像の読み込み
    img = Image.open(image_path)
    img_width, img_height = img.size

    # 出力フォルダを作成
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 切り分け処理
    tile_num = 0
    for y in range(0, img_height, tile_size):
        for x in range(0, img_width, tile_size):
            # 切り出すエリアの定義
            box = (x, y, x + tile_size, y + tile_size)
            tile = img.crop(box)
            
            # タイル画像の保存
            tile_path = os.path.join(output_folder, f"tile_{tile_num}.png")
            tile.save(tile_path)
            tile_num += 1

    print(f"画像の分割が完了しました。{tile_num}枚の画像を保存しました。")


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

def create_black_mask(image_rgb):
    """
    黒色部分以外を1、黒色部分を0とするマスクを生成する関数
    """
    # 各ピクセルが完全な黒かどうかを判定（全てのチャネルが0）
    mask = np.all(image_rgb == [0, 0, 0], axis=-1)
    # 黒色以外の部分を1に、黒色部分を0にする
    inverse_mask = np.logical_not(mask).astype(np.float32)
    return inverse_mask



def makeSkinSeparation(INPUT_DIR,input_image_list,OUTPUT_DIR,vector):
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
    
    melanin_dir = os.path.join(OUTPUT_DIR, "melanin")
    hemoglobin_dir = os.path.join(OUTPUT_DIR, "hemoglobin")
    shading_dir = os.path.join(OUTPUT_DIR, "shading")
    os.makedirs(melanin_dir, exist_ok=True)
    os.makedirs(hemoglobin_dir, exist_ok=True)
    os.makedirs(shading_dir, exist_ok=True)
    
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
        img_mask = create_black_mask(image_rgb)
        img_mask = np.repeat(img_mask[np.newaxis, :, :], 3, axis=0)  # 3チャンネルに拡張
        img_mask2 = (1 / 255) + np.zeros_like(img_mask, dtype=np.float32)

        linearSkin = np.zeros_like(image_rgb, dtype=np.float32).transpose(2, 0, 1)
        S = np.zeros_like(image_rgb, dtype=np.float32).transpose(2, 0, 1)

        skin = image_rgb.transpose(2, 0, 1).astype(np.float32)
        for i in range(3):
            linearSkin[i] = np.power((skin[i] / 2), 1) / 255


        # 濃度空間 (log空間) へ
        S = -np.log(linearSkin + img_mask2) * img_mask

        # 肌色空間の起点を 0 へ
        # 必要に応じて調整するパラメーター
        MinSkin = [0, 0, 0]
        for i in range(3):
            S[i] = S[i] - MinSkin[i]
        
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
                    OUTPUT_DIR =melanin_dir
                    output_path = '{}_2_Mel.png'.format(image_basename)
            elif param_index == 1:
                for chn_index in range(3):
                    L_Obj[chn_index] = hemoglobin[chn_index] * L_Hem
                    OUTPUT_DIR =hemoglobin_dir
                    output_path = '{}_3_Hem.png'.format(image_basename)
                
            else:
                for chn_index in range(3):
                    L_Obj[chn_index] =shading[chn_index] * L_Sha
                    OUTPUT_DIR =shading_dir
                    output_path = '{}_1_Sha.png'.format(image_basename)

            # 可視化して保存する。最大値と最小値で規格化する
            img = L_Obj.transpose(1,2,0)
            
            img_exp = np.clip(np.exp(-img), 0, 1) * img_mask.transpose(1,2,0)
            
            # マスク外を灰色に設定
            gray_value = 192 / 255.0 
            img_exp[img_mask.transpose(1, 2, 0) == 0] = gray_value
            
            # min_v = img_exp[img_mask.transpose(1,2,0)>0.0].min()
            # ef_img = img_exp -min_v
            # max_v = ef_img[img_mask.transpose(1,2,0)>0.0].max()
            # ef_img = (ef_img / max_v)
            ef_img =img_exp
            save_image(os.path.join(OUTPUT_DIR, output_path), ef_img)


def merge_images_to_grid(input_dir, output_path, image_size, grid_size, final_image_size):
    """
    指定されたフォルダ内の画像を結合して、1つの大きな画像に保存する関数。

    Parameters:
        - input_dir (str): 入力画像フォルダのパス
        - output_path (str): 出力画像のパス
        - image_size (int): 各タイル画像のサイズ (幅・高さ)
        - grid_size (int): グリッドの行/列の数 (例: 4なら4x4)
        - final_image_size (int): 結合後の最終画像のサイズ (幅・高さ)
    """
    # 空の大きな画像を作成（黒画像で初期化）
    final_image = Image.new('RGB', (final_image_size, final_image_size))

    # 指定フォルダ内の画像を読み込み、ソート
    image_files = [f for f in natsorted(os.listdir(input_dir)) 
                   if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    # 画像を順番に貼り付け
    for idx, filename in enumerate(image_files):
        img = Image.open(os.path.join(input_dir, filename))
        img = img.resize((image_size, image_size))  # サイズを調整

        # グリッド内の行・列を計算
        row = idx // grid_size
        col = idx % grid_size

        # 大きな画像にタイルを貼り付け
        final_image.paste(img, (col * image_size, row * image_size))

    # 結果を保存
    final_image.save(output_path)
    print(f"結合画像を {output_path} に保存しました。")


if __name__ == '__main__':
    
    current= Path(__file__).resolve()
    DATA_DIR =str(current.parents[1] / "data")
    target_list ='sample2'
    target_dir = DATA_DIR +"\\sample2\\"
    
    # 画像
    image_path = target_dir+"sample.png" 
    
    tiles_folder = target_dir +"output_tiles"  # 分割後の画像を保存するフォルダ
    tile_size = 1024  # 正方形の1辺の長さ

    split_image(image_path, tiles_folder, tile_size)
    
    ## 旧マスターベクトル
    # melanin= [0.087196, 0.471511, 0.877539]
    # hemoglobin = [0.132687, 0.640552, 0.756364]
    ## 新マスターベクトル
    melanin    =[0.2203, 0.4788, 0.8499]
    hemoglobin =[0.4350, 0.6929, 0.5750]
    shading    =[ 1.0000, 1.0000, 1.0000 ]

    vector = [melanin,hemoglobin,shading]
    
    
    OUTPUT_DIR = target_dir +"\\output\\"
    INPUT_DIR  =tiles_folder
    input_image_list = glob.glob(str(INPUT_DIR +"\\**.png"))
    input_image_list=natsorted(input_image_list)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    makeSkinSeparation(INPUT_DIR,input_image_list,OUTPUT_DIR,vector)
        
    
    # フォルダのパスを指定
    #   melanin_dir = os.path.join(OUTPUT_DIR, "melanin")
    # hemoglobin_dir = os.path.join(OUTPUT_DIR, "hemoglobin")
    # shading_dir = os.path.join(OUTPUT_DIR, "shading")
    
    targets_list = ["melanin","hemoglobin","shading"]
    # 画像サイズと結合後の大きな画像のサイズ
    image_size = 1024
    grid_size = 16  # 32×32 = 1024枚
    final_image_size = image_size * grid_size
    
    
    for target in targets_list:
        
        input_target = OUTPUT_DIR +"\\" + target+"\\"
        output_image_path_target =target_dir +"\\output-image-"+target +".png"
        merge_images_to_grid(input_target,output_image_path_target,1024,256,16384)
        

    print('Done.')