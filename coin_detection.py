"""硬貨検出アプリ 段階1: 硬貨の検出 + 穴の有無の判定（改善版 v3）

今回の変更点:
  - 写真(画像ファイル)から硬貨を認識できるようにした (--image)
    ・日本語を含むパスの画像も読み込める
    ・大きな写真は処理しやすい大きさに縮小して処理する (PROCESS_MAX_WIDTH)
    ・sキーで、検出結果の画像をファイルに保存できる（発表資料用）
  - カメラ・録画動画の入力は今までどおり使える

これまでの変更点:
  - 硬貨が検出されない不具合を修正 (detect_coins)
    紙の外側(暗い部分)が二値化で白くなって紙を囲むと、紙の上の硬貨が「内側の輪郭」
    として無視されていた。輪郭の深さを数え、穴(奇数の深さ)だけを除外するようにした。
  - 円形度を凸包の周囲長で計算するように変更（二値画像のギザギザで円形度が下がり、
    正常な硬貨が除外されるのを防ぐ）
  - 白黒の向き(背景が明るいか暗いか)を実行中に切り替えられるようにした (bキー)
  - ウィンドウの大きさを変える方式をやめ、表示する画像そのものを半画面サイズに縮小して表示
  - 大津法の閾値を「画面中央の領域だけ」で計算する機能 (USE_ROI_OTSU)
  - 穴の面積比の下限を 0.01 -> 0.005 に変更

操作キー:
    o : 大津法 (Otsu) に切り替え
    a : 適応的二値化 (Adaptive) に切り替え
    b : 背景が明るい／暗い の設定を切り替え（硬貨が白くならないとき）
    s : 検出結果の画像を保存（写真モードでは写真と同じフォルダに保存）
    q : 終了

表示される青字: HSV値・色判定(COPPER/GOLD/SILVER?)・穴判定

実行例:
    uv run python coin_detection.py                       # Webカメラ（外部カメラ優先、なければ内蔵カメラ）
    uv run python coin_detection.py --video a.mp4         # 録画動画
    uv run python coin_detection.py --image coins.jpg     # 写真
"""

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# =====================================================================
# 設定部分（閾値や判定条件はここにまとめる。環境に合わせて調整する）
# =====================================================================
# カメラは自動で選ぶ: 外部カメラがつながっていれば外部カメラ、なければ内蔵カメラ
EXTERNAL_CAMERA_INDEX = 1        # 外部(USB)カメラの番号
BUILTIN_CAMERA_INDEX = 0         # 内蔵カメラの番号
BLUR_KSIZE = (5, 5)              # ノイズ除去(ガウシアンぼかし)のサイズ（奇数）

# 写真を読み込んだときの、処理用の最大の幅[px]。これより大きい写真は縮小する。
# 面積などの閾値はこのくらいの大きさを前提にしているため。
PROCESS_MAX_WIDTH = 1280

# 背景が明るい(白い紙の上に硬貨)なら True、暗い(黒い布の上)なら False
# ※ 実行中に b キーで切り替えられる
BACKGROUND_IS_BRIGHT = False

# 大津法の閾値を画面中央だけで計算するか（画面の端の影響を減らす）
USE_ROI_OTSU = True
ROI_RANGE = (0.25, 0.75)         # 中央領域（画面の25%〜75%の範囲）

# 適応的二値化のパラメータ
ADAPTIVE_BLOCK_SIZE = 51       # 近傍領域のサイズ（奇数）
ADAPTIVE_C = 5                   # 平均から引く定数

# 二値化後の整形（小さな穴やノイズをつぶす）
MORPH_KERNEL_SIZE = 5

# 硬貨とみなす条件
MIN_AREA = 800                   # 面積の最小値[px]
MAX_AREA = 60000                 # 面積の最大値[px]
MIN_CIRCULARITY = 0.75           # 円形度の最小値（1.0が真円）

# 穴の判定条件
HOLE_MIN_AREA_RATIO = 0.005      # 穴の面積 / 硬貨の面積 の最小値
HOLE_MAX_AREA_RATIO = 0.15       # 同 最大値
HOLE_MAX_CENTER_DIST = 0.3       # 穴の中心と硬貨の中心の距離 / 半径 の最大値

# 表示色 (B, G, R)
COLOR_NO_HOLE = (0, 200, 0)      # 穴なし: 緑
COLOR_HOLE = (0, 140, 255)       # 穴あり: オレンジ

# ウィンドウ配置の設定
SCREEN_MARGIN_BOTTOM = 160       # 画面下(タスクバーなど)＋タイトルバー分に空ける高さ[px]
WINDOW_TOP_Y = 0                 # ウィンドウを置く縦位置[px]


# =====================================================================
# データ構造
# =====================================================================

@dataclass
class Coin:
    """検出した硬貨1枚の情報"""
    center: tuple
    radius: float
    area: float
    circularity: float
    has_hole: bool
    contour: np.ndarray
    denomination: str = "不明"
    diameter_mm: float = 0.0


# =====================================================================
# 画像ファイルの読み書き
# =====================================================================
def load_image(path):
    """写真を読み込む。大きすぎる場合は PROCESS_MAX_WIDTH に縮小する

    cv2.imread は日本語を含むパスを読めないことがあるため、
    いったんバイト列で読んでから cv2.imdecode で画像にしている。
    スマホの写真の向き情報(EXIF)は自動で反映される。
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        return None

    h, w = image.shape[:2]
    if w > PROCESS_MAX_WIDTH:
        scale = PROCESS_MAX_WIDTH / w
        image = cv2.resize(image, (PROCESS_MAX_WIDTH, int(h * scale)),
                           interpolation=cv2.INTER_AREA)
    return image


def save_image(path, image):
    """画像を保存する（日本語を含むパスでも保存できるように imencode を使う）"""
    ok, buf = cv2.imencode(".png", image)
    if ok:
        buf.tofile(str(path))
    return ok


def open_camera(index):
    """指定番号のカメラを開く。開けない・映らないときは None を返す"""
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if cap.isOpened():
        ok, _ = cap.read()
        if ok:
            return cap
    cap.release()
    return None


def open_default_camera():
    """外部カメラがつながっていれば外部カメラ、なければ内蔵カメラを開く。

    戻り値: 開いたカメラ。どちらも開けなければ None
    """
    for index, name in ((EXTERNAL_CAMERA_INDEX, "外部カメラ"),
                        (BUILTIN_CAMERA_INDEX, "内蔵カメラ")):
        cap = open_camera(index)
        if cap is not None:
            print(f"{name}（カメラ番号 {index}）で開きます")
            return cap
    return None


def save_outputs(out_dir, stem, method, bg_bright, result_img):
    """検出結果の画像を、わかりやすい名前で保存する"""
    bg = "bright" if bg_bright else "dark"
    result_path = out_dir / f"{stem}_{method}_{bg}_result.png"
    save_image(result_path, result_img)
    print(f"保存しました: {result_path}")


# =====================================================================
# ウィンドウ配置
# =====================================================================
def get_screen_size():
    """画面の解像度(幅, 高さ)を返す。Windows以外では1920x1080を仮定する"""
    try:
        import ctypes
        user32 = ctypes.windll.user32
        # 拡大率(125%や150%)があっても実際のピクセル数で測るための設定
        user32.SetProcessDPIAware()
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    except Exception:
        return 1920, 1080


def calc_display_size(frame_w, frame_h):
    """表示サイズ(幅, 高さ)と、右側ウィンドウの左端のx座標を返す

    幅は画面の半分。縦が画面に収まらない場合は縦横比を保ったまま縮小する。
    """
    screen_w, screen_h = get_screen_size()
    half_w = screen_w // 2

    scale = half_w / frame_w
    max_h = screen_h - SCREEN_MARGIN_BOTTOM - WINDOW_TOP_Y
    if frame_h * scale > max_h:
        scale = max_h / frame_h          # 縦に収まるよう縮小
    return int(frame_w * scale), int(frame_h * scale), half_w


def show(name, image, size):
    """画像を指定サイズに縮小(拡大)して表示する。検出処理には影響しない"""
    # 二値画像は元の画素をそのまま見たいので INTER_NEAREST を使う
    interp = cv2.INTER_AREA if image.ndim == 3 else cv2.INTER_NEAREST
    cv2.imshow(name, cv2.resize(image, size, interpolation=interp))


# =====================================================================
# 処理関数
# =====================================================================
def preprocess(frame):
    """STEP1: グレースケール化 + ノイズ除去"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, BLUR_KSIZE, 0)


def binarize(gray, method, bg_bright=BACKGROUND_IS_BRIGHT):
    """STEP2: 二値化。method は 'otsu' か 'adaptive'

    硬貨が白(255)、背景が黒(0)になるようにそろえる。
    """
    # 背景が明るいときは、硬貨(暗い側)を白にするため反転する
    thresh_type = cv2.THRESH_BINARY_INV if bg_bright else cv2.THRESH_BINARY

    if method == "otsu":
        if USE_ROI_OTSU:
            # 画面中央の領域だけで大津法の閾値を求め、その閾値を画面全体に適用する
            # （画面の端にある暗い部分などに閾値が引っ張られるのを防ぐ）
            h, w = gray.shape
            lo, hi = ROI_RANGE
            roi = gray[int(h * lo):int(h * hi), int(w * lo):int(w * hi)]
            t, _ = cv2.threshold(roi, 0, 255, thresh_type + cv2.THRESH_OTSU)
            _, binary = cv2.threshold(gray, t, 255, thresh_type)
        else:
            # 大津法: 画像全体でヒストグラムから1つの閾値を自動決定する
            _, binary = cv2.threshold(gray, 0, 255, thresh_type + cv2.THRESH_OTSU)
    else:
        # 適応的二値化: 画素ごとに周囲の明るさから閾値を決める
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresh_type, ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C)

    # 小さなノイズ除去(opening)と、小さな穴の埋め込み(closing)
    kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary


def has_center_hole(index, contours, hierarchy, center, radius, coin_area):
    """硬貨中央の穴を、面積・形・位置から判定する。"""

    for j in range(len(contours)):
        if hierarchy[j][3] != index:
            continue

        cnt = contours[j]
        hole_area = cv2.contourArea(cnt)
        if hole_area <= 0:
            continue

        ratio = hole_area / coin_area
        # 1円などの中央刻印を穴と誤認しないよう、面積比を厳しめにする
        if not (0.012 <= ratio <= 0.08):
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * math.pi * hole_area / (perimeter ** 2)
        if circularity < 0.65:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if w == 0 or h == 0:
            continue
        aspect_ratio = w / h
        if not (0.70 <= aspect_ratio <= 1.30):
            continue

        m = cv2.moments(cnt)
        if m["m00"] == 0:
            continue
        hx = m["m10"] / m["m00"]
        hy = m["m01"] / m["m00"]
        dist = math.hypot(hx - center[0], hy - center[1]) / radius
        if dist > 0.18:
            continue

        return True

    return False


def contour_depth(index, hierarchy):
    """輪郭の深さ(何重の内側にあるか)を返す。0=一番外側、1=その内側の穴、2=穴の中の島…"""
    depth = 0
    parent = hierarchy[index][3]
    while parent != -1:
        depth += 1
        parent = hierarchy[parent][3]
    return depth


def detect_coins(binary):
    """STEP3: 輪郭から硬貨を検出し、穴の有無も調べる"""
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    hierarchy = hierarchy[0]

    coins = []
    for i, cnt in enumerate(contours):
        # 白い塊の外周は深さが偶数(0,2,4...)、その中の黒い穴は奇数(1,3,5...)。
        # 穴は硬貨本体ではないので除外する。（紙が暗い部分に囲まれていて硬貨が
        # 深さ2になるような場合でも、硬貨を見逃さないようにするため）
        if contour_depth(i, hierarchy) % 2 == 1:
            continue

        area = cv2.contourArea(cnt)
        if not (MIN_AREA <= area <= MAX_AREA):
            continue

        # 円形度 = 4π×面積 / 周囲長²（真円で1.0）
        # 輪郭のギザギザで周囲長が長くなり円形度が下がるのを防ぐため、
        # 周囲長は凸包(輪郭を外側からゴムで包んだ形)のものを使う
        perimeter = cv2.arcLength(cv2.convexHull(cnt), True)
        if perimeter == 0:
            continue
        circularity = 4 * math.pi * area / (perimeter ** 2)
        if circularity < MIN_CIRCULARITY:
            continue

        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        hole = has_center_hole(i, contours, hierarchy, (cx, cy), radius, area)
        coins.append(Coin((cx, cy), radius, area, circularity, hole, cnt))
    return coins

def get_coin_color(frame, coin):
    """硬貨の中央寄りの複数領域から、表面色の代表値を取得する。

    外周の背景や中央の刻印・反射の影響を減らすため、
    半径の0.18〜0.62倍の円環部分のHSV中央値を使う。
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    x, y = coin.center
    r_outer = coin.radius * 0.62
    r_inner = coin.radius * 0.18

    x1 = max(0, int(x - r_outer))
    x2 = min(frame.shape[1], int(x + r_outer + 1))
    y1 = max(0, int(y - r_outer))
    y2 = min(frame.shape[0], int(y + r_outer + 1))

    roi = hsv[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0, 0.0, 0.0

    yy, xx = np.ogrid[y1:y2, x1:x2]
    distance2 = (xx - x) ** 2 + (yy - y) ** 2
    mask = (distance2 <= r_outer ** 2) & (distance2 >= r_inner ** 2)

    if not np.any(mask):
        return 0.0, 0.0, 0.0

    pixels = roi[mask]
    # 彩度が極端に低い画素（白飛び・強い反射）を色相の判定から除外
    saturated = pixels[pixels[:, 1] >= 25]
    if len(saturated) >= 10:
        pixels = saturated

    return (
        float(np.median(pixels[:, 0])),
        float(np.median(pixels[:, 1])),
        float(np.median(pixels[:, 2])),
    )


def classify_coins(frame, coins):
    """穴・直径を主軸にし、色は補助情報として硬貨を分類する。"""

    if not coins:
        return "硬貨なし"

    features = []
    for coin in coins:
        hue, saturation, value = get_coin_color(frame, coin)

        # 色は照明の影響を受けるため、分類の決定条件にはしすぎない。
        # 今回の実測では10円もH12/S82となったため、色だけで除外しない。
        # 10円の赤銅色は比較的高い彩度で現れた今回の測定値を基準にする。
        # 100円の弱い色かぶり(H14/S39)を銅色と誤認しにくくする。
        is_copper = (hue <= 18 or hue >= 175) and saturation >= 50
        # 5円の実測(H16/S93)をGOLD候補に含める。
        is_gold = 14 <= hue <= 40 and saturation >= 45

        features.append({
            "coin": coin,
            "hue": hue,
            "saturation": saturation,
            "value": value,
            "is_copper": is_copper,
            "is_gold": is_gold,
        })

    # 穴のある硬貨を縮尺の基準にする。
    # 5円/50円は穴があるため、穴が正しく取れた場合に利用する。
    # 10円候補は色だけでなく直径も使い、強い色かぶりの影響を減らす。
    scale_candidates = []
    for f in features:
        coin = f["coin"]
        diameter_px = coin.radius * 2
        if coin.has_hole:
            if f["is_gold"]:
                scale_candidates.append(diameter_px / 22.0)
            else:
                scale_candidates.append(diameter_px / 21.0)
        elif f["is_copper"]:
            scale_candidates.append(diameter_px / 23.5)

    if not scale_candidates:
        for coin in coins:
            coin.denomination = "不明"
            coin.diameter_mm = 0.0
        return "縮尺未確定（穴のある硬貨または10円候補が必要）"

    px_per_mm = float(np.median(scale_candidates))

    coin_sizes = {
        "1円": 20.0,
        "50円": 21.0,
        "5円": 22.0,
        "100円": 22.6,
        "10円": 23.5,
        "500円": 26.5,
    }

    for f in features:
        coin = f["coin"]
        diameter_mm = (coin.radius * 2) / px_per_mm
        coin.diameter_mm = diameter_mm

        # 穴あり硬貨は5円/50円に限定し、色を補助的に使う。
        # H16/S93の5円をGOLD側に含め、50円との取り違えを減らす。
        if coin.has_hole:
            coin.denomination = "5円" if f["is_gold"] else "50円"
            continue

        # 穴なし硬貨の10円と100円は直径が近いため、色を先に確認する。
        # 今回の実測では10円はH12/S82、誤判定した100円はH45/S14。
        # 赤銅色が明確な場合だけ10円とし、それ以外は1/100/500円から選ぶ。
        if f["is_copper"] and 22.0 <= diameter_mm <= 25.5:
            coin.denomination = "10円"
            continue

        # 銀色候補には10円を含めない。これにより直径測定が23.5mm付近に
        # ずれた100円を、10円と誤判定するのを防ぐ。
        candidates = ("1円", "100円", "500円")
        nearest = min(candidates,
                      key=lambda name: abs(coin_sizes[name] - diameter_mm))
        difference = abs(coin_sizes[nearest] - diameter_mm)

        if difference <= 1.5:
            coin.denomination = nearest
        else:
            coin.denomination = "不明"

    return f"scale: {px_per_mm:.2f} px/mm"


def draw_result(frame, coins, method, status, bg_bright):
    """検出結果と硬貨の種類・合計金額・診断情報を見やすく表示"""

    out = frame.copy()

    values = {
        "1円": 1,
        "5円": 5,
        "10円": 10,
        "50円": 50,
        "100円": 100,
        "500円": 500,
    }

    # 硬貨の種類ごとの表示色（OpenCVはBGR）
    denomination_colors = {
        "1円": (255, 200, 0),
        "5円": (0, 200, 255),
        "10円": (0, 140, 255),
        "50円": (255, 0, 200),
        "100円": (200, 200, 200),
        "500円": (0, 220, 0),
        "不明": (0, 0, 255),
    }

    counts = {name: 0 for name in values}
    counts["不明"] = 0
    total = 0

    # 上部の情報パネル
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (out.shape[1], 105), (20, 20, 20), -1)
    out = cv2.addWeighted(overlay, 0.72, out, 0.28, 0)

    # 硬貨ごとの情報
    for c in coins:
        name = c.denomination

        if name in values:
            counts[name] += 1
            total += values[name]
        else:
            counts["不明"] += 1

        color = denomination_colors.get(name, (0, 0, 255))

        # 輪郭
        cv2.drawContours(out, [c.contour], -1, color, 3)

        x, y = int(c.center[0]), int(c.center[1])
        radius = int(c.radius)

        # 硬貨の種類
        label = name
        if c.diameter_mm > 0:
            label += f"  {c.diameter_mm:.1f}mm"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.85
        thickness = 2

        (text_w, text_h), _ = cv2.getTextSize(
            label, font, font_scale, thickness
        )

        label_x = max(0, min(x - text_w // 2, out.shape[1] - text_w))
        label_y = max(text_h + 8, y - radius - 12)

        # ラベルの背景
        cv2.rectangle(
            out,
            (label_x - 5, label_y - text_h - 5),
            (label_x + text_w + 5, label_y + 5),
            (30, 30, 30),
            -1
        )

        cv2.putText(
            out, label, (label_x, label_y),
            font, font_scale, color, thickness, cv2.LINE_AA
        )

        # HSV診断情報
        hue, saturation, value = get_coin_color(frame, c)

        is_copper = (hue <= 18 or hue >= 175) and saturation >= 50
        is_gold = 14 <= hue <= 40 and saturation >= 45

        if is_copper:
            color_judge = "COPPER"
        elif is_gold:
            color_judge = "GOLD"
        elif saturation < 35:
            color_judge = "SILVER?"
        else:
            color_judge = "OTHER"

        diagnostic = (
            f"H:{hue:.0f} S:{saturation:.0f} "
            f"V:{value:.0f} {color_judge}"
        )

        hole_text = "HOLE: YES" if c.has_hole else "HOLE: NO"

        detail_y = y + radius + 20

        cv2.putText(
            out, diagnostic,
            (max(5, x - 65), detail_y),
            font, 0.48, (255, 255, 0), 1, cv2.LINE_AA
        )

        cv2.putText(
            out, hole_text,
            (max(5, x - 65), detail_y + 22),
            font, 0.48, (255, 255, 0), 1, cv2.LINE_AA
        )

    # 上部の情報表示
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(
        out, f"COINS: {len(coins)}",
        (15, 35), font, 0.85, (255, 255, 255), 2, cv2.LINE_AA
    )

    cv2.putText(
        out, f"TOTAL: {total} YEN",
        (15, 75), font, 1.0, (0, 255, 100), 2, cv2.LINE_AA
    )

    cv2.putText(
        out, f"{status}  MODE: {method.upper()}",
        (out.shape[1] - 380, 35),
        font, 0.55, (255, 255, 255), 1, cv2.LINE_AA
    )

    # 硬貨の内訳
    summary = "  ".join(
        f"{name}:{count}" for name, count in counts.items()
    )

    cv2.putText(
        out, summary,
        (15, out.shape[0] - 18),
        font, 0.55, (255, 255, 255), 1, cv2.LINE_AA
    )

    return out



# =====================================================================
# メイン
# =====================================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="写真(画像ファイル)のパス")
    parser.add_argument("--video", help="録画動画のパス（省略時はWebカメラ）")
    args = parser.parse_args()

    # 入力の準備
    still = None
    cap = None

    if args.image:
        # 写真モード
        still = load_image(args.image)

        if still is None:
            print(f"写真を読み込めませんでした: {args.image}")
            print("パスやファイル名(拡張子を含む)を確認してください。")
            return

        out_dir = Path(args.image).resolve().parent
        stem = Path(args.image).stem

    else:
        # 動画またはWebカメラモード
        if args.video:
            cap = cv2.VideoCapture(args.video)

            if not cap.isOpened():
                print(f"動画を開けませんでした: {args.video}")
                print("ファイルパスを確認してください。")
                cap.release()
                return
        else:
            cap = open_default_camera()

            if cap is None:
                print("カメラを開けませんでした。")
                print("カメラが接続されているか、他のアプリで使用中でないか確認してください。")
                return

        out_dir = Path.cwd()
        stem = "capture"

    # 初期設定
    method = "otsu"
    bg_bright = BACKGROUND_IS_BRIGHT
    fps = 0.0
    display_size = None
    right_x = 0
    first_frame = True

    # メインループ
    while True:
        t0 = time.perf_counter()

        # 画像を取得
        if still is not None:
            frame = still.copy()
        else:
            ok, frame = cap.read()

            if not ok:
                print("映像を取得できませんでした。")
                break

        # 最初の1フレームで表示サイズを決定
        if display_size is None:
            h, w = frame.shape[:2]
            win_w, win_h, right_x = calc_display_size(w, h)
            display_size = (win_w, win_h)

        # 画像処理
        gray = preprocess(frame)
        binary = binarize(gray, method, bg_bright)
        coins = detect_coins(binary)

        # 硬貨の種類を分類
        scale_status = classify_coins(frame, coins)

        # 処理時間・FPSの表示
        if still is not None:
            status = f"time: {(time.perf_counter() - t0) * 1000:.0f} ms"
        else:
            status = f"FPS: {fps:.1f}"

        # 検出結果を描画
        result_img = draw_result(
            frame,
            coins,
            method,
            status + "  " + scale_status,
            bg_bright,
        )

        # 画像を表示
        show("result", result_img, display_size)
        show("binary", binary, display_size)
        # ウィンドウの位置を設定
        if first_frame:
            cv2.waitKey(1)
            cv2.moveWindow("result", 0, WINDOW_TOP_Y)
            cv2.moveWindow("binary", right_x, WINDOW_TOP_Y)
            first_frame = False

        # キー入力
        key = cv2.waitKey(30 if still is not None else 1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("o"):
            method = "otsu"

        elif key == ord("a"):
            method = "adaptive"

        elif key == ord("b"):
            bg_bright = not bg_bright

        elif key == ord("s"):
            save_outputs(
                out_dir,
                stem,
                method,
                bg_bright,
                result_img,
            )

        # FPS計算
        dt = time.perf_counter() - t0

        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt

    # カメラ・動画を解放
    if cap is not None:
        cap.release()

    # ウィンドウを閉じる
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()