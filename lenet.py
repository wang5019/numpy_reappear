from numpy_reappear import block


class Lenet5():
    def __init__(self,rbf_w):
        self.net = [block.Conv2([6,1,5,5]),                     #nx24x24x6
                    block.Pool2([2,2,6], 'mean', 'sigmoid', 2),    #nx12x12x6
                    block.Leconv2(),                                            #nx8x8x16
                    block.Pool2([2,2,16], 'max', 'sigmoid', 2),    #nx4x4x16
                    block.Conv2([120,16,4,4]),                  #nx1x1x120
                    block.Fullycon2([120, 84], 'Letanh'),
                    # bk.Rbfcon2([84, 10], 'relu')]   #nx84
                    block.Rbfcon2(rbf_w)]                                       #nx10
        self.learning_rate = [1e-2,1e-2,1e-3,1e-3,1e-4,1e-4] + [0]   #Rbfcon is not training

    def init_weights(self):
        self.net[0].init_weights('Le')
        self.net[1].init_weights('Le')
        self.net[2].init_weights('Le')
        self.net[3].init_weights('Le')
        self.net[4].init_weights('Le')
        self.net[5].init_weights('Xavier')

    def forward(self, inputs):
        for k, each in enumerate(self.net):
            if k == 0:
                each.forward(inputs)
            else:
                each.forward(self.net[k-1].outputs)
        outputs = self.net[-1].outputs

        return outputs

    def update(self, delta_in):
        idx = [i for i in range(len(self.net))]
        idx.reverse()

        for i in idx:
            if i == len(self.net) - 1:
                self.net[i].update(delta_in, self.learning_rate[i])
            else:
                self.net[i].update(self.net[i+1].delta_out, self.learning_rate[i])

    def train(self, images, labels):
        outputs = self.forward(images)
        loss, delta_in = block.Lossfun(outputs, labels, 'LeLoss')
        acc = block.get_accuracy_lenet(outputs, labels)
        self.update(delta_in)
        return loss, acc

    def test(self, images, labels):
        outputs = self.forward(images)
        acc = block.get_accuracy_lenet(outputs, labels)
        return acc
