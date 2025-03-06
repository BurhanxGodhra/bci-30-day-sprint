
!pip install brainflow scikit-learn scipy
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from scipy import signal
import numpy as np
from sklearn.svm import SVC

def classify_power_features(data, sampling_rate, bands=[(4, 8), (8, 13), (13, 30)]):
    if len(data.shape) == 1:
        data = data.reshape(1, -1)
    powers = []
    for low, high in bands:
        nyquist = sampling_rate / 2
        b, a = signal.butter(2, [low/nyquist, high/nyquist], btype='band')
        filtered = signal.lfilter(b, a, data[0])
        power = np.mean(filtered**2)
        powers.append(power)
    features = np.array(powers).reshape(1, -1)
    X_train = np.array([[0.4, 0.8, 0.2], [0.4, 0.5, 0.4], [0.4, 0.2, 0.8]])
    y_train = np.array([0, 1, 2])  # Rest, focus, move
    clf = SVC(kernel='rbf')
    clf.fit(X_train, y_train)
    pred = clf.predict(features)[0]
    return ["rest", "focus", "move"][pred]

board = BoardShim(BoardIds.SYNTHETIC_BOARD.value, BrainFlowInputParams())
board.prepare_session()
board.start_stream()
import time
time.sleep(5)
data = board.get_board_data()
board.stop_stream()
board.release_session()
state = classify_power_features(data[1], 250)
print(f"Predicted State: {state}")
