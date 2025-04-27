import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from scipy.signal import savgol_filter
import matplotlib.cm as cm
from tqdm import tqdm

# === PARAMÈTRES ===
VIDEO_PATH = '0424.mp4'
OUTPUT_VIDEO = 'output_annotated.mp4'
OUTPUT_CSV = 'bubbles_data.csv'

PIXEL_TO_MM = 0.1  # Exemple : 1 pixel = 0.1 mm
MAX_DISTANCE = 50  # Distance max pour associer une bulle entre 2 frames (en pixels)

# === INITIALISATION ===
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

while True:
    ret, frame = cap.read()
    if not ret:
        break

    output_frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                               param1=50, param2=30, minRadius=5, maxRadius=60)

    detections = []
    if circles is not None:
        circles = np.around(circles[0]).astype(int)
        for x, y, r in circles:
            detections.append({'x': x, 'y': y, 'r': r, 'assigned': False})

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
                speed = (dist * PIXEL_TO_MM) / dt  # mm/s

                data['last_position'] = (d['x'], d['y'])
                data['last_frame'] = frame_idx
                data['r'] = d['r']
                data['speeds'].append(speed)

                data_export.append([frame_idx, bubble_id, d['x'], d['y'], d['r'], speed,time_since_birth])
                time_since_birth = (frame_idx - data['birth_frame']) / fps

                # Normalise le temps de vie entre 0 et 1 (à ajuster selon la durée max que tu veux)
                normalized_age = min(time_since_birth / 3, 1.0)  # ici on suppose 3 secondes max

                # Passe d'une couleur BLEUE (neuve) à ROUGE (vieille)
                r = int(255 * normalized_age)
                g = int(255 * (1 - normalized_age))
                b = 0

                color = (b, g, r)  # OpenCV veut BGR

                # Dessine la bulle avec la couleur correspondante
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
            data_export.append([frame_idx, bubble_id_counter, d['x'], d['y'], d['r'], 0,0])
            cv2.circle(output_frame, (d['x'], d['y']), d['r'], (255, 0, 0), 2)
            cv2.putText(output_frame, f"ID:{bubble_id_counter}", (d['x'] - 10, d['y'] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    out.write(output_frame)
    frame_idx += 1

cap.release()
out.release()

# === EXPORT CSV ===
with open(OUTPUT_CSV, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['frame', 'bubble_id', 'x', 'y', 'radius', 'speed_mm_per_s', 'time_since_birth'])
    writer.writerows(data_export)

# === HISTOGRAMMES ===
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


# === NOUVEAUX GRAPHES : Rayon et Vitesse en fonction du Temps écoulé depuis la naissance ===

# On récupère les données organisées par bulle
bubbles_birth_time = {}  # bubble_id -> birth time (en secondes)
bubbles_radii = {}       # bubble_id -> liste de (frame, radius)
bubbles_speeds = {}      # bubble_id -> liste de (frame, speed)

# Filtrage des bulles qui n'ont pas vécu assez longtemps (par exemple, moins de 5 frames)
min_lifetime = 5  # Nombre minimal de frames pour considérer une bulle

for row in data_export:
    frame, bubble_id, x, y, radius, speed, time_since_birth = row
    if bubble_id not in bubbles_birth_time:
        bubbles_birth_time[bubble_id] = (frame / fps)
        bubbles_radii[bubble_id] = []
        bubbles_speeds[bubble_id] = []
    
    # Ajouter les données seulement si la bulle a vécu assez longtemps
    if frame - (bubbles_birth_time[bubble_id] * fps) >= min_lifetime:
        bubbles_radii[bubble_id].append((frame, radius))
        if speed > 0:
            bubbles_speeds[bubble_id].append((frame, speed))

# Palette de couleurs pour chaque bulle
colors = cm.viridis(np.linspace(0, 1, len(bubbles_birth_time)))

# === AFFICHAGE ===
plt.figure(figsize=(14, 6))

# --- Rayon ---
plt.subplot(1, 2, 1)

all_times_radii = []
all_radii = []

for (bubble_id, color) in zip(bubbles_birth_time.keys(), colors):
    birth_frame = bubbles_birth_time[bubble_id] * fps
    frames_radii = bubbles_radii[bubble_id]
    
    times_since_birth = [(frame - birth_frame) / fps for frame, _ in frames_radii]
    radii_values = [radius for _, radius in frames_radii]
    
    if times_since_birth:  # Vérifie qu'il y a bien des données
        plt.scatter(times_since_birth, radii_values, color=color, alpha=0.6, label=f"Bulle {bubble_id}")
        all_times_radii.extend(times_since_birth)
        all_radii.extend(radii_values)

# Courbe lissée globale
if len(all_times_radii) > 5:  # Besoin d'un minimum de points pour lisser
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

for (bubble_id, color) in zip(bubbles_birth_time.keys(), colors):
    birth_frame = bubbles_birth_time[bubble_id] * fps
    frames_speeds = bubbles_speeds[bubble_id]
    
    times_since_birth = [(frame - birth_frame) / fps for frame, _ in frames_speeds]
    speeds_values = [speed for _, speed in frames_speeds]
    
    if times_since_birth:  # Vérifie qu'il y a bien des données
        plt.scatter(times_since_birth, speeds_values, color=color, alpha=0.6, label=f"Bulle {bubble_id}")
        all_times_speeds.extend(times_since_birth)
        all_speeds.extend(speeds_values)

# Courbe lissée globale
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
