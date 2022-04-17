# Path vers les images à analyser
single_path='captures_calibration/' # images individuelles
stereo_path=single_path # images stéréo (peuvent être les mêmes que individuelles)

# Fichiers de calibration:
left_xml='cam1.xml'
right_xml='cam2.xml'

# Paramètres du damier
patternSize = (7,9)
squareSize = 200

# Path vers les folders où enregistrer les images détectées
single_detected_path='output/singles_detected/' #images détectée lors de la calibration individuelle
stereo_detected_path='output/stereo_detected/'#images détectée lors de la calibration stéréo
