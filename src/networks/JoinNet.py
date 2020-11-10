from base.base_net import BaseNet


class JoinNet(BaseNet):
    """
    JoinNet的网络结构
                            svddNet
        lstmNet Encoder ---|
                            lstmNet Decoder

    要想实现联合训练，必须获取训练好的 lstmNet 的 svddNet
    """
    def __init__(self, lstm_net, svdd_net):
        super().__init__()
        self.lstm_net = lstm_net
        self.svdd_net = svdd_net

    def forward(self, x):
        """ 联合训练 """
        # Ajoy 其实每个程序、每个部分都是分开的，通过encoder可以得到中间编码
        x = self.lstm_net.encode(x)
        # Ajoy 通过decoder可以得到输入数据的重构
        y0 = self.lstm_net.decode(x)
        # Ajoy 相当于forward
        y1 = self.svdd_net.compute(x)
        return y0, y1
