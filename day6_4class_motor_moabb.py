{"cells":[{"cell_type":"code","source":["# MOABB-inspired 4-class motor imagery with CSP + SVM\n","\n","import mne\n","import numpy as np\n","from sklearn.model_selection import train_test_split\n","from sklearn.metrics import accuracy_score\n","from pyriemann.estimation import Covariances\n","from pyriemann.classification import MDM\n","\n","# Load GDF\n","raw = mne.io.read_raw_gdf('/content/A01T.gdf', preload=True)\n","events, event_dict = mne.events_from_annotations(raw)\n","\n","# Preprocessing\n","raw.notch_filter(50, picks='eeg')\n","raw.filter(8, 30, fir_design='firwin', picks='eeg')\n","ica = mne.preprocessing.ICA(n_components=15, random_state=42)\n","ica.fit(raw)\n","raw = ica.apply(raw)\n","\n","# Epoching\n","sfreq = 250\n","tmin, tmax = 1, 6  # Your 33% window\n","n_samples = int((tmax - tmin) * sfreq)\n","event_ids = {'769': 0, '770': 1, '771': 2, '772': 3}\n","\n","X, y = [], []\n","for event in events:\n","    event_id = event[2]\n","    if event_id in [event_dict[str(k)] for k in event_ids]:\n","        start = event[0] - int(4 * sfreq)\n","        stop = start + n_samples\n","        if start >= 0 and stop <= raw.n_times:\n","            data, _ = raw[:, start:stop]\n","            X.append(data)\n","            y.append(event_ids[str([k for k, v in event_dict.items() if v == event_id][0])])\n","\n","X = np.array(X)\n","y = np.array(y)\n","print(f\"X shape: {X.shape}, y shape: {y.shape}\")\n","print(f\"Label counts: {np.unique(y, return_counts=True)}\")\n","\n","# Feature Extraction + MDM\n","cov = Covariances().fit_transform(X)\n","clf = MDM(metric='riemann')\n","X_train, X_test, y_train, y_test = train_test_split(cov, y, test_size=0.3, random_state=42)\n","clf.fit(X_train, y_train)\n","y_pred = clf.predict(X_test)\n","print(f\"Test accuracy: {accuracy_score(y_test, y_pred):.2f}\")"],"metadata":{"colab":{"base_uri":"https://localhost:8080/"},"id":"OH7gOSTLuW3n","executionInfo":{"status":"ok","timestamp":1741285895179,"user_tz":-300,"elapsed":40053,"user":{"displayName":"Burhanuddin Mustafa","userId":"06265806691346669695"}},"outputId":"d95f07cd-217a-4c57-ac36-862c60ff95f1"},"execution_count":8,"outputs":[{"output_type":"stream","name":"stdout","text":["Extracting EDF parameters from /content/A01T.gdf...\n","GDF file detected\n","Setting channel info structure...\n","Could not determine channel type of the following channels, they will be set as EEG:\n","EEG-Fz, EEG, EEG, EEG, EEG, EEG, EEG, EEG-C3, EEG, EEG-Cz, EEG, EEG-C4, EEG, EEG, EEG, EEG, EEG, EEG, EEG, EEG-Pz, EEG, EEG, EOG-left, EOG-central, EOG-right\n","Creating raw.info structure...\n","Reading 0 ... 672527  =      0.000 ...  2690.108 secs...\n"]},{"output_type":"stream","name":"stderr","text":["/usr/lib/python3.11/contextlib.py:144: RuntimeWarning: Channel names are not unique, found duplicates for: {'EEG'}. Applying running numbers for duplicates.\n","  next(self.gen)\n"]},{"output_type":"stream","name":"stdout","text":["Used Annotations descriptions: ['1023', '1072', '276', '277', '32766', '768', '769', '770', '771', '772']\n","Filtering raw data in 1 contiguous segment\n","Setting up band-stop filter from 49 - 51 Hz\n","\n","FIR filter parameters\n","---------------------\n","Designing a one-pass, zero-phase, non-causal bandstop filter:\n","- Windowed time-domain design (firwin) method\n","- Hamming window with 0.0194 passband ripple and 53 dB stopband attenuation\n","- Lower passband edge: 49.38\n","- Lower transition bandwidth: 0.50 Hz (-6 dB cutoff frequency: 49.12 Hz)\n","- Upper passband edge: 50.62 Hz\n","- Upper transition bandwidth: 0.50 Hz (-6 dB cutoff frequency: 50.88 Hz)\n","- Filter length: 1651 samples (6.604 s)\n","\n"]},{"output_type":"stream","name":"stderr","text":["[Parallel(n_jobs=1)]: Done  17 tasks      | elapsed:    0.8s\n"]},{"output_type":"stream","name":"stdout","text":["Filtering raw data in 1 contiguous segment\n","Setting up band-pass filter from 8 - 30 Hz\n","\n","FIR filter parameters\n","---------------------\n","Designing a one-pass, zero-phase, non-causal bandpass filter:\n","- Windowed time-domain design (firwin) method\n","- Hamming window with 0.0194 passband ripple and 53 dB stopband attenuation\n","- Lower passband edge: 8.00\n","- Lower transition bandwidth: 2.00 Hz (-6 dB cutoff frequency: 7.00 Hz)\n","- Upper passband edge: 30.00 Hz\n","- Upper transition bandwidth: 7.50 Hz (-6 dB cutoff frequency: 33.75 Hz)\n","- Filter length: 413 samples (1.652 s)\n","\n"]},{"output_type":"stream","name":"stderr","text":["[Parallel(n_jobs=1)]: Done  17 tasks      | elapsed:    0.8s\n"]},{"output_type":"stream","name":"stdout","text":["Fitting ICA to data using 25 channels (please be patient, this may take a while)\n","Selecting by number: 15 components\n","Fitting ICA took 32.6s.\n","Applying ICA to Raw instance\n","    Transforming to ICA space (15 components)\n","    Zeroing out 0 ICA components\n","    Projecting back using 25 PCA components\n","X shape: (288, 25, 1250), y shape: (288,)\n","Label counts: (array([0, 1, 2, 3]), array([72, 72, 72, 72]))\n","Test accuracy: 0.33\n"]}]}],"metadata":{"colab":{"provenance":[],"authorship_tag":"ABX9TyO1I7lkimGR3XlXbPTv2Xtm"},"kernelspec":{"display_name":"Python 3","name":"python3"},"language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":0}