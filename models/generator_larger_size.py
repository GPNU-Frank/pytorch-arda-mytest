"""Generator for ARDA.

learn the domain-invariant feature representations from inputs across domains.
"""

from torch import nn


class Generator_Larger(nn.Module):
    """LeNet encoder model for ARDA."""

    def __init__(self):
        """Init LeNet encoder."""
        super(Generator_Larger, self).__init__()

        self.restored = False

        self.encoder = nn.Sequential(
            # 1st conv block
            # input [1 x 36 x 36]
            # output [64 x 16 x 16]


            nn.Conv2d(1, 64, 5, 1, 0, bias=False),
            nn.MaxPool2d(2),
            nn.ReLU(),
            # 2nd conv block
            # input [64 x 16 x 16]
            # output [50 x 6 x 6]
            nn.Conv2d(64, 50, 5, 1, 0, bias=False),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        """
        In
        ............................
        Op
        *****.......................
        .*****......................
        .......................*****
        .......................*****

        Out
        + x 24
        24 x 24
        12 x 12
        ->



        """
        self.fc1 = nn.Linear(50 * 6 * 6, 500)

    def forward(self, input):
        """Forward the LeNet."""
        conv_out = self.encoder(input)
        feat = self.fc1(conv_out.view(-1, 50 * 6 * 6))
        # inside:
        # [method-view-call: conv_out, -1, 800]
        return feat
