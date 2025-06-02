import io
import math
import os
import tempfile
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_hex

import streamlit_app as st

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Tracé de route sur carte TEMSI",
    page_icon="✈️",
    layout="wide"
)


def interpolate_latitude_coefficients(target_latitude):
    """Retourne les coefficients a, b, c interpolés pour une latitude."""
    # Données de référence
    latitudes = [50, 45, 40]
    a_vals = [0.0001104939295672841, 0.00009629046695457561, 0.00009412799656861709]
    b_vals = [-0.19883534859359414, -0.17285957127863721, -0.18022927166549757]
    c_vals = [1622.0591758645953, 981.2993561345577, 340.7626029781216]

    # Interpolation quadratique pour chaque coefficient
    a = np.polyval(np.polyfit(latitudes, a_vals, deg=2), target_latitude)
    b = np.polyval(np.polyfit(latitudes, b_vals, deg=2), target_latitude)
    c = np.polyval(np.polyfit(latitudes, c_vals, deg=2), target_latitude)

    return a, b, c


def get_affine_lon_function(lon_deg):
    """Calcule les coefficients a, b de la fonction affine x(y) = a*y + b pour une longitude."""
    # Données de référence
    x0, y0, a_ref = 899.5, 6106.94, -11.41995

    # Cas spécial : longitude 0° (méridien central)
    if lon_deg == 0:
        return 0, x0  # Droite verticale x = x0

    # Calcul des coefficients
    a = a_ref / (lon_deg / 5)
    b = y0 - a * x0

    return a, b


def extract_waypoint_coordinates(file_content):
    """
    Extracts the coordinates of waypoints from a Garmin XML flight plan file content.

    :param file_content: XML content as bytes or string
    :return: A list of lists with waypoint data: [latitude, longitude]
    """
    # Convertir en string si nécessaire
    if isinstance(file_content, bytes):
        try:
            xml_content = file_content.decode('utf-8')
        except UnicodeDecodeError:
            try:
                xml_content = file_content.decode('utf-16')
            except UnicodeDecodeError:
                raise ValueError("Failed to decode the file with supported encodings.")
    else:
        xml_content = file_content

    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        raise ValueError(f"XML parsing error: {e}")

    # Déclaration du namespace Garmin
    namespace = {'ns': 'http://www8.garmin.com/xmlschemas/FlightPlan/v1'}

    # Recherche de tous les waypoints dans <waypoint-table>
    waypoints = []
    for wp in root.findall('.//ns:waypoint-table/ns:waypoint', namespace):
        try:
            lat = float(wp.find('ns:lat', namespace).text)
            lon = float(wp.find('ns:lon', namespace).text)
            waypoints.append([lat, lon])
        except AttributeError:
            # Ignore les entrées incomplètes
            continue

    return waypoints


def intersection_quadratic_affine(a_q, b_q, c_q, a_a, b_a):
    """
    Calcule les points d'intersection entre une fonction quadratique et une affine.
    """
    A = a_q
    B = b_q - a_a
    C = c_q - b_a

    delta = B ** 2 - 4 * A * C

    if delta < 0:
        # Pas d'intersection réelle, en réalité c'est lié à la longitude 0° qui est verticale
        return [(899.5, a_q * 899.5 ** 2 + b_q * 899.5 + c_q)]

    sqrt_delta = math.sqrt(delta)
    x1 = (-B - sqrt_delta) / (2 * A)
    x2 = (-B + sqrt_delta) / (2 * A)

    # On peut utiliser la quadratique pour calculer y
    y1 = a_q * x1 ** 2 + b_q * x1 + c_q
    y2 = a_q * x2 ** 2 + b_q * x2 + c_q

    if delta == 0:
        return [(x1, y1)]
    else:
        return [(x1, y1), (x2, y2)]


def gnss_to_pixel(lat, lon):
    """Convertit des coordonnées GNSS en pixels"""
    a, b, c = interpolate_latitude_coefficients(lat)
    a2, b2 = get_affine_lon_function(lon)
    intersection = intersection_quadratic_affine(a, b, c, a2, b2)
    return intersection[-1] if intersection else None


def create_route_on_map(coordinates, pdf_file=None):
    """
    Crée une image avec les points et la route sur la carte PDF.

    Args:
        coordinates: Liste de tuples (latitude, longitude)
        pdf_file: Fichier PDF de la carte (optionnel)

    Returns:
        Figure matplotlib
    """
    # Convertir les coordonnées GNSS en pixels (avec inversion axe Y)
    pixel_coords = []
    coord_info = []

    for lat, lon in coordinates:
        pixel_pos = gnss_to_pixel(lat, lon)
        if pixel_pos:
            pixel_coords.append((pixel_pos[0], 1785 - pixel_pos[1]))  # Inversion axe Y
            coord_info.append(f"(lat={lat:.4f}, lon={lon:.4f}) → Pixel: ({pixel_pos[0]:.1f}, {pixel_pos[1]:.1f})")

    if not pixel_coords:
        st.error("Aucune coordonnée valide trouvée")
        return None, []

    fig, ax = plt.subplots(1, 1, figsize=(15, 10))

    # Charger l'image PDF si fournie
    if pdf_file is not None:
        try:
            # Sauvegarder temporairement le fichier PDF uploadé
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf_file.getvalue())
                tmp_path = tmp_file.name

            from pdf2image import convert_from_path
            pages = convert_from_path(tmp_path, dpi=200)
            img_array = np.array(pages[0])
            ax.imshow(img_array, extent=[0, 2526, 1785, 0])

            # Nettoyer le fichier temporaire
            os.unlink(tmp_path)

        except ImportError:
            st.warning("pdf2image non installé. Affichage d'un rectangle simulant la carte...")
            from matplotlib.patches import Rectangle
            rect = Rectangle((0, 0), 2526, 1785, linewidth=2, edgecolor='black', facecolor='lightgray', alpha=0.3)
            ax.add_patch(rect)
            ax.set_xlim(0, 2526)
            ax.set_ylim(0, 1785)
        except Exception as e:
            st.warning(f"Erreur lors du chargement du PDF: {e}")
            from matplotlib.patches import Rectangle
            rect = Rectangle((0, 0), 2526, 1785, linewidth=2, edgecolor='black', facecolor='lightgray', alpha=0.3)
            ax.add_patch(rect)
            ax.set_xlim(0, 2526)
            ax.set_ylim(0, 1785)
    else:
        # Créer un fond gris si pas de PDF
        from matplotlib.patches import Rectangle
        rect = Rectangle((0, 0), 2526, 1785, linewidth=2, edgecolor='black', facecolor='lightgray', alpha=0.3)
        ax.add_patch(rect)
        ax.set_xlim(0, 2526)
        ax.set_ylim(0, 1785)

    # Dessiner la route avec gradient de couleur
    x_coords = [coord[0] for coord in pixel_coords]
    y_coords = [coord[1] for coord in pixel_coords]

    n_segments = len(pixel_coords) - 1
    if n_segments > 0:
        for i in range(n_segments):
            t = i / n_segments
            r = int(255 * t)
            g = int(255 * (1 - t))
            b = 0
            color = to_hex((r / 255, g / 255, b / 255))
            x_segment = [pixel_coords[i][0], pixel_coords[i + 1][0]]
            y_segment = [pixel_coords[i][1], pixel_coords[i + 1][1]]
            ax.plot(x_segment, y_segment, color=color, linewidth=3, alpha=0.7)

        # Marquer le début et la fin
        ax.plot(x_coords[0], y_coords[0], 'go', markersize=10, label='Départ')
        ax.plot(x_coords[-1], y_coords[-1], 'ro', markersize=10, label='Arrivée')

        # Légende pour les couleurs
        ax.plot([], [], color='#00FF00', linewidth=3, alpha=0.7, label='Départ (vert)')
        ax.plot([], [], color='#FF0000', linewidth=3, alpha=0.7, label='Arrivée (rouge)')

    # Supprimer axes et quadrillage
    ax.axis('off')

    # Garder la légende
    ax.legend(loc='upper right', framealpha=0.9, fontsize=10)

    plt.tight_layout()

    return fig, coord_info


# Interface Streamlit
def main():
    st.title("✈️ Tracé de route sur carte TEMSI")
    st.markdown("---")

    # Sidebar pour les fichiers
    st.sidebar.header("📁 Fichiers d'entrée")

    # Upload du fichier PDF
    pdf_file = st.sidebar.file_uploader(
        "Carte TEMSI (PDF)",
        type=['pdf'],
        help="Fichier PDF de la carte TEMSI"
    )

    # Upload du fichier de plan de vol
    fpl_file = st.sidebar.file_uploader(
        "Plan de vol ForeFlight (.fpl)",
        type=['fpl', 'xml'],
        help="Fichier XML contenant les waypoints du plan de vol"
    )

    # Colonnes principales
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📍 Coordonnées extraites")

        if fpl_file is not None:
            try:
                # Extraire les coordonnées
                coordinates = extract_waypoint_coordinates(fpl_file.getvalue())

                if coordinates:
                    st.success(f"✅ {len(coordinates)} waypoints trouvés")

                    # Afficher les coordonnées dans un tableau
                    import pandas as pd
                    df = pd.DataFrame(coordinates, columns=['Latitude', 'Longitude'])
                    df.index = df.index + 1
                    df.index.name = 'Waypoint'
                    st.dataframe(df, use_container_width=True)

                    # Bouton pour générer la carte
                    if st.button("🗺️ Générer la carte avec tracé", type="primary"):
                        with st.spinner("Génération de la carte en cours..."):
                            fig, coord_info = create_route_on_map(coordinates, pdf_file)

                            if fig is not None:
                                with col2:
                                    st.subheader("🗺️ Carte avec tracé")
                                    st.pyplot(fig)

                                    # Bouton de téléchargement
                                    buf = io.BytesIO()
                                    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                                    buf.seek(0)

                                    st.download_button(
                                        label="💾 Télécharger la carte (PNG)",
                                        data=buf.getvalue(),
                                        file_name="carte_route_temsi.png",
                                        mime="image/png"
                                    )

                                # Afficher les détails de conversion
                                st.subheader("🔧 Détails de conversion")
                                with st.expander("Voir les conversions GNSS → Pixel"):
                                    for info in coord_info:
                                        st.text(info)
                else:
                    st.warning("⚠️ Aucun waypoint trouvé dans le fichier")

            except Exception as e:
                st.error(f"❌ Erreur lors du traitement du fichier: {str(e)}")
        else:
            st.info("📄 Veuillez télécharger un fichier de plan de vol (.fpl)")

    with col2:
        if fpl_file is None:
            st.subheader("🗺️ Aperçu")
            st.info("La carte avec le tracé apparaîtra ici une fois les fichiers chargés")

    # Section d'aide
    st.markdown("---")
    with st.expander("ℹ️ Aide et informations"):
        st.markdown("""
        ### Comment utiliser cette application :

        1. **Téléchargez votre carte TEMSI** (PDF) dans la barre latérale
        2. **Téléchargez votre plan de vol** (.fpl) ForeFlight 
        3. **Cliquez sur "Générer la carte"** pour voir le tracé
        4. **Téléchargez le résultat** au format PNG

        ### Formats supportés :
        - **Carte TEMSI** : fichiers PDF
        - **Plan de vol** : fichiers .fpl (XML ForeFlight)

        ### Fonctionnalités :
        - ✅ Conversion automatique GNSS → pixels
        - ✅ Tracé coloré du départ (vert) à l'arrivée (rouge)
        - ✅ Marqueurs de début/fin de route
        - ✅ Export haute qualité (PNG 300 DPI)

        ### Note technique :
        L'application utilise des coefficients d'interpolation spécifiques aux cartes TEMSI 
        pour convertir les coordonnées GPS en positions pixels sur la carte.
        """)


if __name__ == "__main__":
    main()