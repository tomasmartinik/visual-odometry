import numpy as np
import cv2 as cv
import argparse

parser = argparse.ArgumentParser(description='This sample demonstrates Lucas-Kanade Optical Flow calculation.')
parser.add_argument('--image', type=str, default='data/video_01.MOV', help='cesta k souboru s obrazem')
parser.add_argument('--chunk_size', type=int, default=100, help='Velikost každé části videa')
args = parser.parse_args()

cap = cv.VideoCapture(args.image)
fps = cap.get(cv.CAP_PROP_FPS)  # Získání počtu snímků za sekundu
frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))  # Celkový počet snímků ve videu

# Inicializace proměnných pro odometrii
odometry = np.zeros(2)  # Pro každý bod ukládáme změnu pozice (X, Y)
total_distance = 0  # Celková ujetá vzdálenost

pixel_to_cm_ratio = 20 / 480  # Předpokládaný poměr pixelů k centimetrům, upravte podle skutečného poměru

# Inicializace pro zobrazení středu pohybu kamery
center_trajectory = []

# První snímek
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Část videa
chunk_start = 0
chunk_end = min(args.chunk_size, frame_count)

while chunk_start < frame_count:
    # Nastavení načítání části videa
    cap.set(cv.CAP_PROP_POS_FRAMES, chunk_start)

    while cap.get(cv.CAP_PROP_POS_FRAMES) < chunk_end:
        ret, frame = cap.read()
        if not ret:
            print('Konec videa!')
            break
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Výpočet optického toku
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, winSize=(15, 15), maxLevel=2,
                                              criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

        # Vyfiltrované body
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # Ověření, že jsou dostupné body pro výpočet odometrie
        if len(good_new) < 2:
            print('Ztracené body, pokračuji dále.')
        else:
            # Průměrný pohyb bodů
            avg_movement = np.mean(good_new - good_old, axis=0)

            # Přičítání průměrného pohybu k odometrii
            odometry += avg_movement

            # Výpočet celkové ujeté vzdálenosti
            total_distance += np.linalg.norm(avg_movement)

            # Uložení středu pohybu kamery pro zobrazení
            center_trajectory.append(np.round(np.mean(good_new, axis=0)).astype(int).tolist())  # Přidána konverze na seznam

            # Aktualizace snímku a bodů pro další iteraci
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

            # Zobrazení výsledků, můžete tuto část přizpůsobit svým potřebám
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                frame = cv.circle(frame, (int(a), int(b)), 5, (0, 255, 0), -1)

            # Zobrazení středu pohybu kamery
            for point in center_trajectory:
                frame = cv.circle(frame, tuple(point), 2, (0, 0, 255), -1)  # Přidána konverze na tuple

            cv.imshow('frame', frame)
            k = cv.waitKey(30) & 0xff
            if k == 27:
                break

    # Přesun na další část videa
    chunk_start = chunk_end
    chunk_end = min(chunk_end + args.chunk_size, frame_count)

# Zavření okna s videem
cv.destroyAllWindows()

# Výsledná odometrie a celková ujetá vzdálenost jsou uloženy v proměnných 'odometry' a 'total_distance'
print("Délka videa: {} sekund".format(frame_count / fps))
print("Celková ujetá vzdálenost: {} cm".format(total_distance * pixel_to_cm_ratio))
