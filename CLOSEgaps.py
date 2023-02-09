import torch
import torch.nn as nn
import torch_geometric.nn as hnn

class CLOSEgaps(nn.Module):
    def __init__(self, input_num, input_feature_num, emb_dim, conv_dim, head=3, p=0.1, L=1,
                 use_attention=True, similarity=None):
        super(CLOSEgaps, self).__init__()
        self.emb_dim = emb_dim
        self.conv_dim = conv_dim
        self.p = p
        self.input_num = input_num
        self.head = head
        self.hyper_conv_L = L
        self.linear_encoder = nn.Linear(input_feature_num, emb_dim)
        self.similarity_liner = nn.Linear(input_num, emb_dim)
        self.max_pool = hnn.global_max_pool
        self.similarity = similarity
        self.in_channel = emb_dim
        if similarity is not None:
            self.in_channel = 2 * emb_dim

        self.relu = nn.ReLU()
        self.hypergraph_conv = hnn.HypergraphConv(self.in_channel, conv_dim, heads=head, use_attention=use_attention,
                                                  dropout=p)
        if L > 1:
            self.hypergraph_conv_list = []
            for l in range(L - 1):
                self.hypergraph_conv_list.append(
                    hnn.HypergraphConv(head * conv_dim, conv_dim, heads=head, use_attention=use_attention, dropout=p))

        if use_attention:
            self.hyper_attr_liner = nn.Linear(input_num, self.in_channel)
            if L > 1:
                self.hyperedge_attr_list = []
                for l in range(L - 1):
                    self.hyperedge_attr_list.append(nn.Linear(input_num, head * conv_dim))
        self.hyperedge_linear = nn.Linear(conv_dim * head, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_fetures, incidence_matrix):
        input_nodes_features = self.relu(self.linear_encoder(input_fetures))
        if self.similarity is not None:
            simi_feature = self.relu(self.similarity_liner(self.similarity))
            input_nodes_features = torch.cat((simi_feature, input_nodes_features), dim=1)

        row, col = torch.where(incidence_matrix.T)
        edges = torch.cat((col.view(1, -1), row.view(1, -1)), dim=0)
        hyperedge_attr = self.hyper_attr_liner(incidence_matrix.T)
        input_nodes_features = self.hypergraph_conv(input_nodes_features, edges, hyperedge_attr=hyperedge_attr)
        if self.hyper_conv_L > 1:
            for l in range(self.hyper_conv_L - 1):
                layer_hyperedge_attr = self.hyperedge_attr_list[l](incidence_matrix.T)
                input_nodes_features = self.hypergraph_conv_list[l](input_nodes_features, edges,
                                                                    hyperedge_attr=layer_hyperedge_attr)
                input_nodes_features = self.relu(input_nodes_features)

        hyperedge_feature = torch.mm(incidence_matrix.T, input_nodes_features)
        return self.hyperedge_linear(hyperedge_feature)

    def predict(self, input_fetures, incidence_matrix):
        return self.softmax(self.forward(input_fetures, incidence_matrix))