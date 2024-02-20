import numpy as np
import cv2 as cv
import argparse

parser = argparse.ArgumentParser(description='This sample demonstrates Lucas-Kanade Optical Flow calculation.')
parser.add_argument('--image', type=str, default='data/video_01.MOV', help='cesta k souboru s obrazem')
args = parser.parse_args()

cap = cv.VideoCapture(args.image)
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
color = np.random.randint(0, 255, (100, 3))

# První snímek
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Inicializace proměnných pro odometrii
odometry = np.zeros(2)  # Pro každý bod ukládáme změnu pozice (X, Y)
total_distance = 0  # Celková ujetá vzdálenost

while True:
    ret, frame = cap.read()
    if not ret:
        print('Nebyly zachyceny žádné snímky!')
        break
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Výpočet optického toku
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Vyfiltrované body
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # Ověření, že jsou dostupné body pro výpočet odometrie
    if len(good_new) < 2:
        break

    # Průměrný pohyb bodů
    avg_movement = np.mean(good_new - good_old, axis=0)

    # Přičítání průměrného pohybu k odometrii
    odometry += avg_movement

    # Výpočet celkové ujeté vzdálenosti
    total_distance += np.linalg.norm(avg_movement)

    # Aktualizace snímku a bodů pro další iteraci
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    # Zobrazení výsledků, můžete tuto část přizpůsobit svým potřebám
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)

    cv.imshow('frame', frame)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

# Zavření okna s videem
cv.destroyAllWindows()

# Výsledná odometrie a celková ujetá vzdálenost jsou uloženy v proměnných 'odometry' a 'total_distance'
print("Odometrie:")
print(odometry)
print("Celková ujetá vzdálenost:")
print(total_distance)
