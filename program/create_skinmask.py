import cv2
import numpy as np
import mediapipe as mp

# MediaPipeの顔ランドマーク検出モジュールを初期化
mp_face_mesh = mp.solutions.face_mesh

def extract_skin_in_face_contour(input_path, output_path):
    # 画像を読み込む
    image = cv2.imread(input_path)
    if image is None:
        print("画像の読み込みに失敗しました。パスを確認してください。")
        return

    # BGRからRGBに変換
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # MediaPipeの顔ランドマーク検出の初期化
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        # 顔ランドマークの検出
        results = face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            print("顔が検出されませんでした。")
            return

        # 画像と同じサイズの黒いマスクを作成
        mask = np.zeros_like(image)

        # 顔の輪郭（ランドマーク）を取得
        for face_landmarks in results.multi_face_landmarks:
            points = [(int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])) 
                      for landmark in face_landmarks.landmark]

            # ポリゴンで輪郭を描画
            hull = cv2.convexHull(np.array(points))
            cv2.fillConvexPoly(mask, hull, (255, 255, 255))

        # マスクで顔の部分を抽出
        face_region = cv2.bitwise_and(image, mask)

        # 顔領域をHSV色空間に変換
        hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)

        # 肌色のHSV範囲を定義
        lower_skin = np.array([0, 48, 80], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)

        # 肌色のマスクを作成
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # 肌色部分のみを抽出
        skin_only = cv2.bitwise_and(face_region, face_region, mask=skin_mask)

        # 結果画像を保存
        cv2.imwrite(output_path, skin_only)
        print(f"結果の画像が {output_path} に保存されました。")

# 使用例
input_image_path = "c:\\Users\\kine0\\tumuraLabo\\M1\\SIE\\SkinSeparation-lab\\kinemasa-Skin-separation-tecnology\\data\\SIE-raw\\png\\Cam_23_1.png"  # 入力画像のパス
output_image_path = "result.png"  # 出力画像のパス

extract_skin_in_face_contour(input_image_path, output_image_path)