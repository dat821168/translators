from .nmt import NMTModel
from translators.cores.modules import RREncoder, RRDecoder, Embedder


class GNMT(NMTModel):
    def __init__(self, config):
        super(GNMT, self).__init__()

        embedder = Embedder(config=config)

        self.encoder = RREncoder(config=config, embedder=embedder)

        self.decoder = RRDecoder(config=config, embedder=embedder)

    def forward(self, src, src_len, tgt):
        context = self.encode(src, src_len)
        context = (context, src_len, None)
        output, _, _ = self.decode(tgt, context, False)
        return output
