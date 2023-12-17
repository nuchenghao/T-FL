import argparse


def args_client():
    parser = argparse.ArgumentParser()
    parser.add_argument('--server_ip', type=str, required=True, help="Ip of server")
    parser.add_argument('--server_port', type=int, required=True, help="Port of server")
    args = parser.parse_args()
    return args


def args_server():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, required=True, help="listening host")
    parser.add_argument('--port', type=int, required=True, help="listening port")
    parser.add_argument('--numClient', type=int, default=1, help="The number of Clients")
    parser.add_argument('--numGlobalTrain', type=int, default=30, help="The number of Global training")
    parser.add_argument("--numLocalTrain", type=int, default=5, help="The number of local training")
    parser.add_argument("--batchSize", type=int, default=256, help="The batch size")
    parser.add_argument("--learningRate", type=float, default=0.01, help="The learning rate")
    args = parser.parse_args()
    return args