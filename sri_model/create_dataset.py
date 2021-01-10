import numpy as np


class kddcup99:
    """
    Class for the kddcup99 intrusion detection dataset.
    """

    def converter(self, s):
        if (s == b'icmp'): return 0.0
        elif (s == b'tcp'): return 1.0
        elif (s == b'udp'): return 2.0
        elif (s == b'IRC'): return 0.0
        elif (s == b'X11'): return 1.0
        elif (s == b'Z39_50'): return 2.0
        elif (s == b'aol'): return 3.0
        elif (s == b'atuh'): return 4.0
        elif (s == b'bgp'): return 5.0
        elif (s == b'courier'): return 6.0
        elif (s == b'csnet_ns'): return 7.0
        elif (s == b'ctf'): return 8.0
        elif (s == b'daytime'): return 9.0
        elif (s == b'discard'): return 10.0
        elif (s == b'domain'): return 11.0
        elif (s == b'domain_u'): return 12.0
        elif (s == b'echo'): return 13.0
        elif (s == b'eco_i'): return 14.0
        elif (s == b'ecr_i'): return 15.0
        elif (s == b'efs'): return 16.0
        elif (s == b'exec'): return 17.0
        elif (s == b'finger'): return 18.0
        elif (s == b'ftp'): return 19.0
        elif (s == b'ftp_data'): return 20.0
        elif (s == b'gopher'): return 21.0
        elif (s == b'harvest'): return 22.0
        elif (s == b'hostname'): return 23.0
        elif (s == b'http'): return 24.0
        elif (s == b'http_278'): return 25.0
        elif (s == b'http_443'): return 26.0
        elif (s == b'http_800'): return 27.0
        elif (s == b'imap4'): return 28.0
        elif (s == b'iso_tsap'): return 29.0
        elif (s == b'klogin'): return 30.0
        elif (s == b'kshell'): return 31.0
        elif (s == b'ldap'): return 32.0
        elif (s == b'link'): return 33.0
        elif (s == b'login'): return 34.0
        elif (s == b'mtp'): return 35.0
        elif (s == b'name'): return 36.0
        elif (s == b'netbios_'): return 37.0
        elif (s == b'netstat'): return 38.0
        elif (s == b'nnsp'): return 39.0
        elif (s == b'nntp'): return 40.0
        elif (s == b'ntp_u'): return 41.0
        elif (s == b'other'): return 42.0
        elif (s == b'pm_dump'): return 43.0
        elif (s == b'pop_2'): return 44.0
        elif (s == b'pop_3'): return 45.0
        elif (s == b'printer'): return 46.0
        elif (s == b'private'): return 47.0
        elif (s == b'red_i'): return 48.0
        elif (s == b'remote_j'): return 49.0
        elif (s == b'rje'): return 50.0
        elif (s == b'shell'): return 51.0
        elif (s == b'smtp'): return 52.0
        elif (s == b'sql_net'): return 53.0
        elif (s == b'ssh'): return 54.0
        elif (s == b'sunrpc'): return 55.0
        elif (s == b'supdup'): return 56.0
        elif (s == b'systat'): return 57.0
        elif (s == b'telnet'): return 58.0
        elif (s == b'tftp_u'): return 59.0
        elif (s == b'tim_i'): return 60.0
        elif (s == b'time'): return 61.0
        elif (s == b'urh_i'): return 62.0
        elif (s == b'urp_i'): return 63.0
        elif (s == b'uucp'): return 64.0
        elif (s == b'uucp_pat'): return 65.0
        elif (s == b'vmnet'): return 66.0
        elif (s == b'whois'): return 67.0
        elif (s == b'OTH'): return 0.0
        elif (s == b'REJ'): return 1.0
        elif (s == b'RSTO'): return 2.0
        elif (s == b'RSTR'): return 3.0
        elif (s == b'S0'): return 4.0
        elif (s == b'S1'): return 5.0
        elif (s == b'S2'): return 6.0
        elif (s == b'S3'): return 7.0
        elif (s == b'SF'): return 8.0
        elif (s == b'Sh'): return 9.0
        elif (s == b'back.'): return 0.0
        elif (s == b'buffer_overflow.'): return 1.0
        elif (s == b'ftp_write.'): return 2.0
        elif (s == b'guess_passwd.'): return 3.0
        elif (s == b'imap.'): return 4.0
        elif (s == b'ipsweep.'): return 5.0
        elif (s == b'land.'): return 6.0
        elif (s == b'loadmodule.'): return 7.0
        elif (s == b'multihop.'): return 8.0
        elif (s == b'neptune.'): return 9.0
        elif (s == b'nmap.'): return 10.0
        elif (s == b'normal.'): return 11.0
        elif (s == b'perl.'): return 12.0
        elif (s == b'phf.'): return 13.0
        elif (s == b'pod.'): return 14.0
        elif (s == b'portsweep.'): return 15.0
        elif (s == b'rootkit.'): return 16.0
        elif (s == b'satan.'): return 17.0
        elif (s == b'smurf.'): return 18.0
        elif (s == b'spy.'): return 19.0
        elif (s == b'teardrop.'): return 20.0
        elif (s == b'warezclient.'): return 21.0
        elif (s == b'warezmaster.'): return 22.0
        else: return -1.0

    def __init__(self):
        # Load kddcup99 data
        print("loading data")
        data = np.loadtxt("/home/srinivas/Desktop/kddcup.data_10_percent_corrected", delimiter=",", converters={1: self.converter, 2: self.converter, 3: self.converter, 41: self.converter}, dtype=np.float32)
        np.save("/home/srinivas/Desktop/sri_model/dataset/data.npy", data)
        # data = np.load("../Datasets/KDDcup99_data/data.npy")
        # self.train = (data[494021:, 0:41], data[494021:, 41])
        # self.eval = (data[:494021, 0:41], data[:494021, 41])

    def get_dataset_name(self):
        return "KDDcup99"

k=kddcup99()




