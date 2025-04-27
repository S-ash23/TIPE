import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from scipy.signal import savgol_filter
import matplotlib.cm as cm
from tqdm import tqdm

# === PARAMÈTRES ===
VIDEO_PATH = '20250327_170621.mp4'
OUTPUT_VIDEO = 'output_annotated.mp4'
OUTPUT_CSV = 'bubbles_data.csv'
MAX_DISTANCE = 50  # Distance max pour associer une bulle entre 2 frames (en pixels)

# === FONCTIONS ===

def select_scale_and_crop(cap):
    print("Sélectionnez 2 points pour définir l'échelle (pixels -> mm)")
    scale_points = []
    crop_points = []
    scale_done = False
    crop_done = False

    def select_points(event, x, y, flags, param):
        nonlocal scale_points, crop_points, scale_done, crop_done
        # Sélection des points pour l'échelle
        if not scale_done:
            if event == cv2.EVENT_LBUTTONDOWN:
                scale_points.append((x, y))
                print(f"Point sélectionné pour l'échelle: ({x}, {y})")
                if len(scale_points) == 2:
                    scale_done = True
                    print("Échelle définie. Maintenant sélectionnez un cadre (rectangle) pour recadrer.")
        # Sélection du cadre pour le recadrage
        elif not crop_done:
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(crop_points) == 0:  # Premier point du cadre
                    crop_points.append((x, y))
                    print(f"Premier point sélectionné pour le recadrage: ({x}, {y})")
                elif len(crop_points) == 1:  # Deuxième point du cadre
                    crop_points.append((x, y))
                    crop_done = True
                    print(f"Second point sélectionné pour le recadrage: ({x}, {y})")
                    print("Recadrage défini.")

    ret, first_frame = cap.read()
    if not ret:
        print("Erreur lors de la lecture de la vidéo.")
        return None, None
    clone = first_frame.copy()
    cv2.namedWindow("Sélection")
    cv2.setMouseCallback("Sélection", select_points)

    while True:
        display = clone.copy()

        # Affichage des points pour l'échelle
        for p in scale_points:
            cv2.circle(display, p, 5, (0, 0, 255), -1)

        # Affichage du rectangle de recadrage "en construction"
        if len(crop_points) == 1:
            cv2.circle(display, crop_points[0], 5, (255, 0, 0), -1)  # Affiche le premier point
            # Affichage du rectangle en cours, en fonction de la position de la souris
            cv2.rectangle(display, crop_points[0], (x, y), (0, 255, 0), 2)  # Affiche un rectangle partiel

        elif len(crop_points) == 2:
            cv2.rectangle(display, crop_points[0], crop_points[1], (0, 255, 0), 2)  # Affiche le rectangle final

        cv2.imshow("Sélection", display)
        key = cv2.waitKey(1)

        # Sortir de la boucle avec 'Esc' si l'utilisateur ne veut pas continuer
        if key == 27:  # 'Esc' key
            print("Fermeture de la sélection.")
            cv2.destroyAllWindows()
            return None, None

        # Si l'échelle et le recadrage sont faits, on quitte la boucle
        if scale_done and crop_done:
            break

    cv2.destroyAllWindows()

    # Vérification que l'utilisateur a sélectionné deux points pour le recadrage
    if len(crop_points) != 2:
        print("Erreur : Vous devez sélectionner deux points pour le recadrage.")
        return None, None
    else:
        print(f"Recadrage sélectionné : {crop_points}")

    # Calcul de l'échelle (pixel -> mm)
    pixel_distance = np.linalg.norm(np.array(scale_points[0]) - np.array(scale_points[1]))
    real_distance_mm = float(input("Entrez la distance réelle (en mm) entre les 2 points : "))
    pixel_to_mm = real_distance_mm / pixel_distance

    # Définition des coordonnées du recadrage
    x_min = min(crop_points[0][0], crop_points[1][0])
    y_min = min(crop_points[0][1], crop_points[1][1])
    x_max = max(crop_points[0][0], crop_points[1][0])
    y_max = max(crop_points[0][1], crop_points[1][1])

    print(f"Coordonnées du recadrage : ({x_min}, {y_min}, {x_max}, {y_max})")

    return (x_min, y_min, x_max, y_max), pixel_to_mm


def process_video(cap, crop_coords, pixel_to_mm):
    (x_min, y_min, x_max, y_max) = crop_coords
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (x_max - x_min, y_max - y_min))

    bubbles = {}
    data_export = []
    bubble_id_counter = 0
    frame_idx = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    for _ in tqdm(range(total_frames), desc="Traitement vidéo"):
        ret, frame = cap.read()
        if not ret:
            break

        frame = frame[y_min:y_max, x_min:x_max]
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
                    speed = (dist * pixel_to_mm) / dt

                    data['last_position'] = (d['x'], d['y'])
                    data['last_frame'] = frame_idx
                    data['r'] = d['r']
                    data['speeds'].append(speed)

                    data_export.append([frame_idx, bubble_id, d['x'], d['y'], d['r'], speed, time_since_birth])

                    normalized_age = min(time_since_birth / 3, 1.0)
                    r = int(255 * normalized_age)
                    g = int(255 * (1 - normalized_age))
                    color = (0, g, r)

                    cv2.circle(output_frame, (d['x'], d['y']), d['r'], color, 2)
                    break

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

        out.write(output_frame)
        frame_idx += 1

    cap.release()
    out.release()
    return data_export

def export_csv(data_export, filename):
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'bubble_id', 'x', 'y', 'radius', 'speed_mm_per_s', 'time_since_birth'])
        writer.writerows(data_export)

def plot_histograms(data_export):
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

def plot_time_graphs(data_export):
    fps = 30  # Adapte si besoin
    bubbles_birth_time = {}
    bubbles_radii = {}
    bubbles_speeds = {}

    min_lifetime = 5  # frames

    for row in data_export:
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

    # Rayon
    plt.subplot(1, 2, 1)
    all_times_radii = []
    all_radii = []
    for (bubble_id, color) in zip(bubbles_birth_time.keys(), colors):
        birth_frame = bubbles_birth_time[bubble_id] * fps
        frames_radii = bubbles_radii[bubble_id]
        times_since_birth = [(frame - birth_frame) / fps for frame, _ in frames_radii]
        radii_values = [radius for _, radius in frames_radii]
        if times_since_birth:
            plt.scatter(times_since_birth, radii_values, color=color, alpha=0.6)
            all_times_radii.extend(times_since_birth)
            all_radii.extend(radii_values)

    if len(all_times_radii) > 5:
        idx_sort = np.argsort(all_times_radii)
        all_times_radii = np.array(all_times_radii)[idx_sort]
        all_radii = np.array(all_radii)[idx_sort]
        smoothed_radii = savgol_filter(all_radii, window_length=5, polyorder=3)
        plt.plot(all_times_radii, smoothed_radii, color='black', lw=2)

    plt.title('Rayon en fonction du temps depuis la naissance')
    plt.xlabel('Temps (s)')
    plt.ylabel('Rayon (pixels)')

    # Vitesse
    plt.subplot(1, 2, 2)
    all_times_speeds = []
    all_speeds = []
    for (bubble_id, color) in zip(bubbles_birth_time.keys(), colors):
        birth_frame = bubbles_birth_time[bubble_id] * fps
        frames_speeds = bubbles_speeds[bubble_id]
        times_since_birth = [(frame - birth_frame) / fps for frame, _ in frames_speeds]
        speeds_values = [speed for _, speed in frames_speeds]
        if times_since_birth:
            plt.scatter(times_since_birth, speeds_values, color=color, alpha=0.6)
            all_times_speeds.extend(times_since_birth)
            all_speeds.extend(speeds_values)

    if len(all_times_speeds) > 5:
        idx_sort = np.argsort(all_times_speeds)
        all_times_speeds = np.array(all_times_speeds)[idx_sort]
        all_speeds = np.array(all_speeds)[idx_sort]
        smoothed_speeds = savgol_filter(all_speeds, window_length=5, polyorder=3)
        plt.plot(all_times_speeds, smoothed_speeds, color='black', lw=2)

    plt.title('Vitesse en fonction du temps depuis la naissance')
    plt.xlabel('Temps (s)')
    plt.ylabel('Vitesse (mm/s)')

    plt.tight_layout()
    plt.savefig('graphiques_vs_timesincebirth.png')
    plt.show()

# === MAIN ===
if __name__ == "__main__":
    cap = cv2.VideoCapture(VIDEO_PATH)
    crop_coords, pixel_to_mm = select_scale_and_crop(cap)
    data_export = process_video(cap, crop_coords, pixel_to_mm)
    export_csv(data_export, OUTPUT_CSV)
    plot_histograms(data_export)
    plot_time_graphs(data_export)
    print("✅ Programme terminé avec succès !")
