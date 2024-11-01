from PIL import Image

def crop_and_concat(input_path, output_path, points, size):
    # 画像を読み込む
    image = Image.open(input_path)
    
    # 切り出した正方形の数だけ横に並べた新しい画像を作成
    num_points = len(points)
    new_width = size * num_points  # 横幅は (size × 切り出し点の数)
    new_height = size  # 高さは正方形の1辺の長さと同じ
    new_image = Image.new("RGB", (new_width, new_height))

    # 各点から正方形を切り出し、横に並べて結合
    for i, (x, y) in enumerate(points):
        cropped_image = image.crop((x, y, x + size, y + size))
        new_image.paste(cropped_image, (i * size, 0))  # 横方向に配置

    # 結合した画像を保存
    new_image.save(output_path)
    print(f"結合された画像が {output_path} に保存されました。")

# 使用例
input_image_path = "c:\\Users\\kine0\\tumuraLabo\\M1\\SIE\\SkinSeparation-lab\\kinemasa-Skin-separation-tecnology\\data\\SIE-raw\\png-auto-bright\\Cam_23_1.png"  # 入力画像のパス
output_image_path = "auto-bright-regular2.png"  # 出力画像のパス
points = [(1200,3000),(2300,3200),(1600,1800),(2400,2750)]  # 切り出し開始点のリスト patch3
# points = [(1100,2800)] ##Noshadow4.
# points = [(1300,2900)] ##mini.

size = 300  # 切り出す正方形のサイズ

crop_and_concat(input_image_path, output_image_path, points, size)