
import cv2
from scipy.optimize import minimize
import numpy as np
from scipy.optimize import fmin
import plotly.graph_objects as go
from  pathlib import Path
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


## ==============================================================
##  最適化アルゴリズム
##===============================================================
def fmin_Make_M(K, res):
    M = np.empty([K, K])
    for m1 in range(0, K):
        for m2 in range(0, K - m1):
            E12 = np.mean((res[0, :] ** m1) * (res[1, :] ** m2))
            E1E2 = np.mean(res[0, :] ** m1) * np.mean(res[1, :] ** m2)
            M[m1, m2] = E12 - E1E2
    return M


def factT(x):
    countfactT = 1
    for ifactT in range(1, x + 1):
        countfactT = countfactT * ifactT
    return countfactT


def GfuncT(ga1, ga2, gb1, gb2):
    Sigma = 1
    G = 1
    if (ga1 + gb1) % 2 == 0:
        k = (ga1 + gb1) // 2
        J2k = (factT(2 * k) * (2 * np.pi) ** (1 / 2)) / (
            ((4**k) * factT(k)) * (Sigma ** (2 * k - 1))
        )
        sg = ((-1) ** ((ga1 - gb1) / 2) * J2k) / (factT(ga1) * factT(gb1))
        G = G * sg
    else:
        G = 0
    if (ga2 + gb2) % 2 == 0:
        k = (ga2 + gb2) // 2
        J2k = (factT(2 * k) * (2 * np.pi) ** (1 / 2)) / (
            ((4**k) * factT(k)) / (Sigma ** (2 * k - 1))
        )
        sg = ((-1) ** ((ga2 - gb2) / 2) * J2k) / (factT(ga2) * factT(gb2))
        G = G * sg
    else:
        G = 0
    return G


def fmin_Cal_Cost_Burel(K, M):
    CostGMM = 0
    for a1 in range(0, K):
        for a2 in range(0, K - a1):
            for b1 in range(0, K):
                for b2 in range(0, K - b1):
                    CostGMM = CostGMM + GfuncT(a1, a2, b1, b2) * M[a1, a2] * M[b1, b2]
    return CostGMM


def f_burel(s):
    x1 = np.cos(s[0])
    y1 = np.sin(s[0])
    x2 = np.cos(s[1])
    y2 = np.sin(s[1])
    H = np.array([[x1, y1], [x2, y2]])
    res = H @ sensor
    K = 4
    M = fmin_Make_M(K, res)
    ret = fmin_Cal_Cost_Burel(K, M)
    return ret

if __name__ == "__main__":


    # Master Vector provided by Prof. Tsumura
    # melanin_vector	= [0.087196, 0.471511, 0.877539]
    # hemoglobin_vector	= [0.132687, 0.640552, 0.756364]

    # ===============================================================================
    # 画像の読み込み
    # ===============================================================================
    # 陰を含まない領域画像名の入力
    current= Path(__file__)
    DATA_DIR =str(current.parents[1] / "data")
    NShadow_path = DATA_DIR +"/master/Patch1_NoShadow.png"
    NShadow_rgb = cv2.imread(NShadow_path)
    
    Nheight , Nwidth,Nchannel = NShadow_rgb.shape
    Nshadow_r = NShadow_rgb[:,:,2]
    Nshadow_g = NShadow_rgb[:,:,1]
    Nshadow_b = NShadow_rgb[:,:,0]
    
    
    # # 小領域画像名の入力（陰影があってもよい）
    SkinImage_path = DATA_DIR +"/master/Patch1_RegularSkin.png"
    rgb = cv2.imread(SkinImage_path)
    height ,width,channel = rgb.shape
    r = rgb[:,:,2]
    g = rgb[:,:,1]
    b = rgb[:,:,0]
    
    # # # ===============================================================================
    # # # 独立成分分析による色素成分色ベクトルの推定
    # # # ===============================================================================

    # # # 画像空間から濃度空間へ変換
    # # # 肌色ベクトル
    NShadow_logr = -np.log(Nshadow_r/255)
    NShadow_logg = -np.log(Nshadow_g/255)
    NShadow_logb = -np.log(Nshadow_b/255)
    Nskin = np.array([NShadow_logr.flatten(),NShadow_logg.flatten(),NShadow_logb.flatten()])
    
    Nskin_mean = Nskin.mean(axis=1)
    NSkin_MeanMat = np.kron(Nskin_mean[:,np.newaxis], np.ones((1, Nheight * Nwidth)))## 平均値の平面を出している
    Nskin_base = Nskin-NSkin_MeanMat
    # # 濃度ベクトルSの固有値と固有ベクトルを計算    
    # # 1.共分散を求める
    # # 2.固有値を求める eigenvalue 固有ベクトル　eigenvector
    N_covariance = np.cov(Nskin)
    N_eig, N_eigvec = np.linalg.eig(N_covariance)
    
    N_eig = N_eig
    N_eigvec = N_eigvec
    
    # 昇順に並べ替え
    Nidx = np.argsort(N_eig)[::-1]
    N_eig_sorted = N_eig[Nidx]
    N_eigvec_T = N_eigvec.T ##順番入れ替えるときに行と列を対応させるため転置
    
    N_eigvec_sorted =N_eigvec_T[Nidx]
    N_pca1 = N_eigvec_sorted[0,:] 
    N_pca2 = N_eigvec_sorted[1,:] 

    # # ===============================================================================
    # # 独立成分分析による色素成分色ベクトルの推定
    # # ===============================================================================
    # # 画像空間から濃度空間へ変換
    # # 肌色ベクトル
    logr = -np.log(r/255)
    logg = -np.log(g/255)
    logb = -np.log(b/255)
    Skin = np.array([logr.flatten(),logg.flatten(),logb.flatten()])
    ## ベクトル
    vec = np.zeros((3,3))
    vec[0,:] = [1,1,1]
    vec[1,:] = N_pca1
    vec[2,:] = N_pca2 
    # # 肌色分布平面の法線 = 2つの色ベクトルの外積
    # # 平面から法線を求める式(vec(0,:) = [1 1 1]なので式には考慮せず)
    housen = [
        vec[1, 1] * vec[2, 2] - vec[1, 2] * vec[2, 1],
        vec[1, 2] * vec[2, 0] - vec[1, 0] * vec[2, 2],
        vec[1, 0] * vec[2, 1] - vec[1, 1] * vec[2, 0],
    ]
    # # 照明ムラ方向(陰影）と平行な成分をとおる直線と肌色分布平面との交点を求める
    # # housen：肌色分布平面の法線
    # # S：濃度空間でのRGB
    # # vec：独立成分ベクトル
    t = -(housen[0] * Skin[0] + housen[1] * Skin[1] + housen[2] * Skin[2]) / (
        housen[0] * 1 + housen[1] * 1 + housen[2] * 1
    )
    # # 陰影除去
    # # skin_flat：陰影除去したメラヘモベクトルの平面 
    # # rest：陰影成分 (t'*vec(1,:))' + S
    Skin_flat = t+Skin
    # # ===============================================================================
    # # 肌色分布平面上のデータを，主成分分析により白色化する．
    # # ===============================================================================
    # #
    # # ゼロ平均計算用
    Skin_mean = Skin_flat.mean(axis=1)
    Skin_MeanMat = np.kron(Skin_mean[:,np.newaxis], np.ones((1, height * width)))## 平均値の平面を出している

    # # 濃度ベクトルSの固有値と固有ベクトルを計算
    Covariance= np.cov(Skin_flat) ## 陰影除去平面の共分散　＝＞固有値もとめる　ｐ＝直行
    
    eig,eigvec = np.linalg.eig(Covariance)

    # 昇順に並べ替え
    idx = np.argsort(eig)[::-1]
    eig_sorted = eig[idx]
    eigvec_T = eigvec.T ##並び替えのため転地
    eigvec_sorted =eigvec_T[idx]

    ## 第一主成分,第二主成分を格納する行列
    P1P2_vec = eigvec_sorted[0:2,:]

    # # 第1主成分，第2主成分
    Skin_base = Skin_flat - Skin_MeanMat
    Pcomponent = P1P2_vec @ (Skin_flat - Skin_MeanMat) ##平均を引いて正規化　ベクトルかけて白色化
    
    # # ===============================================================================
    # # 独立成分分析
    # # ===============================================================================

    size_dimention, num_samples = Pcomponent.shape

    Pstd = np.std(Pcomponent, axis=1,ddof=1)
    NM = np.diag(1 / Pstd)
    sensor = NM @ Pcomponent  ## 規格化
    np.random.seed(seed=5)
    res = sensor
    flag_1 =0
    flag_2 =0
    while True:
        # Burelの独立評価値をNelder-Mead法で最小化
        # 観測値は相関がある　＝＞　独立なベクトルを求める
        # s   乱数
        # ans: 独立評価値
        s = np.zeros((1,2))
        s[0,0] = np.random.rand(1) * np.pi ##角度でベクトルを表している
        s[0,1] = np.random.rand(1) * np.pi ##角度でベクトルを表している
        s_solved = fmin(f_burel, s) ## 最適化関数
        # s_solved= minimize(f_burel,s,method='Nelder-Mead')
        
        
        ans = f_burel((s_solved)) ##最適化したもの用いて評価値を出す
        ## 角度から座標への変換
        x1 = np.cos(s_solved[0])
        y1 = np.sin(s_solved[0])
        x2 = np.cos(s_solved[1])
        y2 = np.sin(s_solved[1])
        
        
        H = np.array([[x1, y1], [x2, y2]])
        ## unitから　逆算的にメラニン　ヘモグロビンのベクトルを求めている
        unit1 = np.array([1.0, 0.0]).T
        unit2 = np.array([0.0, 1.0]).T

        TM = H @ NM @ P1P2_vec
        
        InvTM = np.linalg.pinv(TM)
        
        # メラニン・ヘモグロビンの色素成分色ベクトル
        c_1 = InvTM @ unit1
        c_2 = InvTM @ unit2
        
        
        ## 例外処理　(反転)
        for i in range(3):
            if c_1[i] < 0:
                c_1[i] *= -1
                flag_1+=1
            if c_2[i] < 0:
                c_2[i] *= -1
                flag_2+=1
        
        
        ## メラニンとヘモグロビンのベクトルの関係性からどっちベクトルか推定
        c_1_norm = c_1 / ((np.sum(c_1**2)) ** (0.5))
        c_2_norm = c_2 / ((np.sum(c_2**2)) ** (0.5))
        
        if c_1_norm[1] < c_1_norm[2]:
            melanin = c_1_norm
            hemoglobin = c_2_norm
        else:
            melanin = c_2_norm
            hemoglobin = c_1_norm

        if np.all(melanin > 0) and np.all(hemoglobin > 0):
            break
        else:
            print("エラー：色ベクトルが負の値です．\n")
        
    ## -------------------------------------------------------------------------
    # print(f'{ans=}')
    
    ## melanin - hemogrobin ベクトル
    
    color_vector = np.zeros((2,3))
    color_vector[0] = hemoglobin
    color_vector[1] = melanin
    
    print(f"{melanin=}")
    print(f"{hemoglobin=}")
    print("color vector estimation is Done.")
