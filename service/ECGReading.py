from pyecg import ECGRecord

class ECGReading:
    def __init__(self):
        self.LEAD_NAME = "ECG"
    def read_ecg(self, ecg_file):
        # Caso necessite ler os sinais em Herz
        # m=np.fromfile(ecg_file,dtype=float)

        record = ECGRecord.from_wfdb(ecg_file)
        signal = record.get_lead(self.LEAD_NAME)
        return signal
        