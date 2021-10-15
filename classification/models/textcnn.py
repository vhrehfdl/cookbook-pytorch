
class TextCNN(nn.Module):
    def __init__(self, hidden_dim, n_vocab, embed_dim, n_classes, word_embeddings):
        super(TextCNN, self).__init__()
        num_channels = 100
        kernel_size = [3, 4, 5]
        max_sen_len = 50
        dropout_keep = 0.8

        self.embeddings = nn.Embedding(n_vocab, embed_dim)
        self.embeddings.weight = nn.Parameter(word_embeddings, requires_grad=False)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim, out_channels=num_channels, kernel_size=kernel_size[0]),
            nn.ReLU(),
            nn.MaxPool1d(max_sen_len - kernel_size[0] + 1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim, out_channels=num_channels, kernel_size=kernel_size[1]),
            nn.ReLU(),
            nn.MaxPool1d(max_sen_len - kernel_size[1] + 1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim, out_channels=num_channels,
                      kernel_size=kernel_size[2]),
            nn.ReLU(),
            nn.MaxPool1d(max_sen_len - kernel_size[2] + 1)
        )

        self.dropout = nn.Dropout(dropout_keep)
        self.fc = nn.Linear(num_channels * len(kernel_size), n_classes)
        self.softmax = nn.Softmax()

    def forward(self, x):
        # x.shape = (max_sen_len, batch_size)
        embedded_sent = self.embeddings(x).permute(0, 2, 1)
        # embedded_sent.shape = (batch_size=64,embed_size=300,max_sen_len=20)

        conv_out1 = self.conv1(embedded_sent).squeeze(2)  # shape=(64, num_channels, 1) (squeeze 1)
        conv_out2 = self.conv2(embedded_sent).squeeze(2)
        conv_out3 = self.conv3(embedded_sent).squeeze(2)

        all_out = torch.cat((conv_out1, conv_out2, conv_out3), 1)
        final_feature_map = self.dropout(all_out)
        final_out = self.fc(final_feature_map)

        return self.softmax(final_out)