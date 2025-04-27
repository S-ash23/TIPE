import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from scipy.signal import savgol_filter
import matplotlib.cm as cm
from tqdm import tqdm

# === PARAMÈTRES ===
print("🔧 Initialisation des paramètres...")
VIDEO_PATH = '0424.mp4'
OUTPUT_VIDEO = 'output_annotated.mp4'
OUTPUT_CSV = 'bubbles_data.csv'

PIXEL_TO_MM = 0.1
MAX_DISTANCE = 50

# === INITIALISATION ===
print("🎥 Chargement de la vidéo...")
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

bubble_id_counter = 0
bubbles = {}  # id: {last_position, last_frame, radius, vitesses[]}
data_export = []

frame_idx = 0
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"✅ Vidéo chargée ({frame_count} frames, {fps:.2f} FPS).")

print("🚀 Début du traitement des frames...")

for frame_idx in tqdm(range(frame_count), desc="Traitement des frames", unit="frame"):
    ret, frame = cap.read()
    if not ret:
        break

    output_frame = frame.copy()
    # Amélioration de l'image avant la détection des cercles
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    # Application d'un flou gaussien pour réduire le bruit
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Détection des cercles avec HoughCircles
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                                param1=100, param2=20, minRadius=10, maxRadius=60)

    # Filtrage des cercles détectés
    detections = []
    if circles is not None:
       circles = np.around(circles[0]).astype(int)
       for x, y, r in circles:
            # Filtrage basé sur le rayon des cercles
            if r < 10 or r > 50:  # Vous pouvez ajuster ces valeurs selon votre besoin
                continue
            detections.append({'x': x, 'y': y, 'r': r, 'assigned': False})

    # Affichage des cercles détectés pour vérification
    for d in detections:
        cv2.circle(output_frame, (d['x'], d['y']), d['r'], (0, 255, 0), 2)  # Cercle vert

    # === ASSOCIATION BULLES ===
    for bubble_id, data in bubbles.items():
        if data['last_frame'] < frame_idx - 1:
            continue
        for d in detections:
            if d['assigned']:
                continue
            dx = d['x'] - data['last_position'][0]
            dy = d['y'] - data['last_position'][1]
            dist_sq = dx * dx + dy * dy
            if dist_sq < MAX_DISTANCE * MAX_DISTANCE:
                d['assigned'] = True
                dt = (frame_idx - data['last_frame']) / fps
                time_since_birth = (frame_idx - data['birth_frame']) / fps
                dist = np.sqrt(dist_sq)
                speed = (dist * PIXEL_TO_MM) / dt

                data['last_position'] = (d['x'], d['y'])
                data['last_frame'] = frame_idx
                data['r'] = d['r']
                data['speeds'].append(speed)

                data_export.append([frame_idx, bubble_id, d['x'], d['y'], d['r'], speed, time_since_birth])

                normalized_age = min(time_since_birth / 3, 1.0)
                r = int(255 * normalized_age)
                g = int(255 * (1 - normalized_age))
                b = 0
                color = (b, g, r)

                cv2.circle(output_frame, (d['x'], d['y']), d['r'], color, 2)
                text = f'ID {bubble_id} {time_since_birth:.2f}s'
                cv2.putText(output_frame, text, (d['x'] - 10, d['y'] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                break

    # === NOUVELLES BULLES ===
    for d in detections:
        if not d['assigned']:
            bubble_id_counter += 1
            bubbles[bubble_id_counter] = {
                'birth_frame': frame_idx,
                'last_position': (d['x'], d['y']),
                'last_frame': frame_idx,
                'r': d['r'],
                'speeds': []
            }
            data_export.append([frame_idx, bubble_id_counter, d['x'], d['y'], d['r'], 0, 0])
            cv2.circle(output_frame, (d['x'], d['y']), d['r'], (255, 0, 0), 2)
            cv2.putText(output_frame, f"ID:{bubble_id_counter}", (d['x'] - 10, d['y'] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    out.write(output_frame)
    frame_idx += 1
    if frame_idx % 100 == 0:
        print(f"🖼️  {frame_idx}/{frame_count} frames traitées...")

cap.release()
out.release()
print("✅ Vidéo annotée sauvegardée.")

# === EXPORT CSV ===
print("💾 Export des données CSV...")
with open(OUTPUT_CSV, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['frame', 'bubble_id', 'x', 'y', 'radius', 'speed_mm_per_s', 'time_since_birth'])
    writer.writerows(data_export)
print(f"✅ Données exportées vers {OUTPUT_CSV}.")

# === HISTOGRAMMES ===
print("📊 Génération des histogrammes...")

speeds = [row[5] for row in data_export if row[5] > 0]
radii = [row[4] for row in data_export]

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(speeds, bins=20, color='skyblue')
plt.title('Distribution des vitesses (mm/s)')
plt.xlabel('Vitesse (mm/s)')
plt.ylabel('Fréquence')

plt.subplot(1, 2, 2)
plt.hist(radii, bins=20, color='salmon')
plt.title('Distribution des rayons (pixels)')
plt.xlabel('Rayon')
plt.ylabel('Fréquence')

plt.tight_layout()
plt.savefig('histogrammes.png')
plt.show()
print("✅ Histogrammes sauvegardés sous histogrammes.png.")

# === NOUVEAUX GRAPHES : Rayon et Vitesse en fonction du Temps écoulé depuis la naissance ===
print("📈 Préparation des graphiques Rayon/Vitesse vs Temps écoulé...")

bubbles_birth_time = {}
bubbles_radii = {}
bubbles_speeds = {}

min_lifetime = 5

for row in tqdm(data_export, desc="Organisation des données"):
    frame, bubble_id, x, y, radius, speed, time_since_birth = row
    if bubble_id not in bubbles_birth_time:
        bubbles_birth_time[bubble_id] = (frame / fps)
        bubbles_radii[bubble_id] = []
        bubbles_speeds[bubble_id] = []

    if frame - (bubbles_birth_time[bubble_id] * fps) >= min_lifetime:
        bubbles_radii[bubble_id].append((frame, radius))
        if speed > 0:
            bubbles_speeds[bubble_id].append((frame, speed))

colors = cm.viridis(np.linspace(0, 1, len(bubbles_birth_time)))

plt.figure(figsize=(14, 6))

# --- Rayon ---
plt.subplot(1, 2, 1)    
all_times_radii = []
all_radii = []

print("📈 Tracé des rayons...")

for (bubble_id, color) in tqdm(zip(bubbles_birth_time.keys(), colors), total=len(bubbles_birth_time), desc="Rayons"):
    birth_frame = bubbles_birth_time[bubble_id] * fps
    frames_radii = bubbles_radii[bubble_id]

    times_since_birth = [(frame - birth_frame) / fps for frame, _ in frames_radii]
    radii_values = [radius for _, radius in frames_radii]

    if times_since_birth:
        plt.scatter(times_since_birth, radii_values, color=color, alpha=0.6, label=f"Bulle {bubble_id}")
        all_times_radii.extend(times_since_birth)
        all_radii.extend(radii_values)
        print(times_since_birth)#Debug pour voir les valeurs de temps depuis la naissance

if len(all_times_radii) > 5:
    idx_sort = np.argsort(all_times_radii)
    all_times_radii = np.array(all_times_radii)[idx_sort]
    all_radii = np.array(all_radii)[idx_sort]

    smoothed_radii = savgol_filter(all_radii, window_length=5, polyorder=3)
    plt.plot(all_times_radii, smoothed_radii, color='black', lw=2, label='Courbe lissée')

plt.title('Rayon en fonction du temps écoulé depuis la naissance')
plt.xlabel('Temps depuis la naissance (s)')
plt.ylabel('Rayon (pixels)')
plt.legend()

# --- Vitesse ---
plt.subplot(1, 2, 2)
all_times_speeds = []
all_speeds = []

print("📈 Tracé des vitesses...")

for (bubble_id, color) in tqdm(zip(bubbles_birth_time.keys(), colors), total=len(bubbles_birth_time), desc="Vitesses"):
    birth_frame = bubbles_birth_time[bubble_id] * fps
    frames_speeds = bubbles_speeds[bubble_id]

    times_since_birth = [(frame - birth_frame) / fps for frame, _ in frames_speeds]
    speeds_values = [speed for _, speed in frames_speeds]

    if times_since_birth:
        plt.scatter(times_since_birth, speeds_values, color=color, alpha=0.6, label=f"Bulle {bubble_id}")
        all_times_speeds.extend(times_since_birth)
        all_speeds.extend(speeds_values)

if len(all_times_speeds) > 5:
    idx_sort = np.argsort(all_times_speeds)
    all_times_speeds = np.array(all_times_speeds)[idx_sort]
    all_speeds = np.array(all_speeds)[idx_sort]

    smoothed_speeds = savgol_filter(all_speeds, window_length=5, polyorder=3)
    plt.plot(all_times_speeds, smoothed_speeds, color='black', lw=2, label='Courbe lissée')

plt.title('Vitesse en fonction du temps écoulé depuis la naissance')
plt.xlabel('Temps depuis la naissance (s)')
plt.ylabel('Vitesse (mm/s)')
plt.legend()

plt.tight_layout()
plt.savefig('graphiques_vs_timesincebirth_tous_points_lisses.png')
plt.show()

print("✅ Graphiques Rayon/Vitesse vs Temps écoulé sauvegardés.")
