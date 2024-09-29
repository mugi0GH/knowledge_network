class FeedForward(nn.module):
    
    
    def __init__(self,d_model,d_ff=2048,dropout=0.1):
        super().__init__()

        # d_ff默认设置为2048
        self.linear_1 = nn.Linear(d_model,d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff,d_model)

    def forward(self,x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x