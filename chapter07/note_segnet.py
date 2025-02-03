def forward(self, x): # torch.Size([b, 3, 256, 256])
    indices = []
    # [torch.Size([b, 64, 256, 256]), torch.Size([b, 128, 128, 128]), torch.Size([b, 256, 64, 64]), torch.Size([b, 512, 32, 32]), torch.Size([b, 512, 16, 16])]
    sizes = []
    # encode block1: torch.Size([b, 3, 128, 128])
    # encode block2: torch.Size([b, 256, 64, 64])
    # encode block3: torch.Size([b, 512, 32, 32])
    # encode block4: torch.Size([b, 512, 16, 16])
    # encode block5: torch.Size([b, 512, 4, 4])
    for enc_block in self.encoder:
        x = enc_block[:-1](x)      # 先执行卷积和激活操作
        sizes.append(x.size())     # 记录输入到池化前的尺寸
        x, ind = enc_block[-1](x)  # 执行池化操作并获取池化索引
        indices.append(ind)        # 保存索引
    # decode block5: torch.Size([b, 512, 16, 16])
    # decode block4: torch.Size([b, 512, 32, 32])
    # decode block3: torch.Size([b, 256, 64, 64])
    # decode block2: torch.Size([b, 3, 128, 128])
    # decode block1: torch.Size([b, 3, 256, 256])
    for dec_block, ind, size in zip(self.decoder, reversed(indices), reversed(sizes)):
        x = dec_block[0](x, ind, output_size=size)
        x = dec_block[1:](x)
    return x