import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierFilter(nn.Module):
    """
    Fourier Filter: to time-variant and time-invariant term
    """
    def __init__(self, mask_spectrum):
        super(FourierFilter, self).__init__()
        self.mask_spectrum = mask_spectrum
        
    def forward(self, x):
        if 'mps' in str(x.device):
            device = 'mps'
            x = x.to(torch.device("cpu"))
        else:
            device = None

        xf = torch.fft.rfft(x, dim=1)
        mask = torch.ones_like(xf)
        mask[:, self.mask_spectrum, :] = 0
        
        x_var = torch.fft.irfft(xf*mask, dim=1)
        x_inv = x - x_var
        
        if device == 'mps':
            x_var = x_var.to('mps')
            x_inv = x_inv.to('mps')
        return x_var, x_inv
    

class FourierSelector(nn.Module):
    def __init__(self, input_len, alpha):
        super(FourierSelector, self).__init__()
        
        self.frequency_size = input_len//2 + 1 
        self.hidden_size_factor = 1
        self.scale = 0.02
        self.alpha = alpha
        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.frequency_size, self.frequency_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor))

        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor, self.frequency_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.frequency_size))

        self.w3 = nn.Parameter(self.scale * torch.randn(2, self.frequency_size, self.frequency_size * self.hidden_size_factor))
        self.b3 = nn.Parameter(self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor))

    def forward(self, x):
        if "mps" in str(x.device):
            device = 'mps'
            x = x.to(torch.device('cpu'))
        else:
            device = None

        xf = torch.fft.rfft(x, dim=1)
        xf = xf.permute(0, 2, 1)

        o1_real = F.relu(torch.einsum('bli,ii->bli', xf.real, self.w1[0]) - torch.einsum('bli,ii->bli', xf.imag, self.w1[1]) + self.b1[0])
        o1_imag = F.relu(torch.einsum('bli,ii->bli', xf.imag, self.w1[0]) + torch.einsum('bli,ii->bli', xf.real, self.w1[1]) + self.b1[1])
        z1 = torch.stack([o1_real, o1_imag], dim=-1)
        # z1 = F.softshrink(z1, lambd=self.sparsity_threshold)

        # o2_real = F.relu(torch.einsum('bli,ii->bli', o1_real, self.w2[0]) - torch.einsum('bli,ii->bli', o1_imag, self.w2[1]) + self.b2[0])
        # o2_imag = F.relu(torch.einsum('bli,ii->bli', o1_imag, self.w2[0]) + torch.einsum('bli,ii->bli', o1_real, self.w2[1]) + self.b2[1])
        # z2 = torch.stack([o2_real, o2_imag], dim=-1)
        # z2 = F.softshrink(z2, lambd=self.sparsity_threshold)

        # o3_real = F.relu(torch.einsum('bli,ii->bli', o2_real, self.w3[0]) - torch.einsum('bli,ii->bli', o2_imag, self.w3[1]) + self.b3[0])
        # o3_imag = F.relu(torch.einsum('bli,ii->bli', o2_imag, self.w3[0]) + torch.einsum('bli,ii->bli', o2_real, self.w3[1]) + self.b3[1])
        # z3 = torch.stack([o3_real, o3_imag], dim=-1)
        # z3 = F.softshrink(z3, lambd=self.sparsity_threshold)

        logits = torch.view_as_complex(z1)
        logits = abs(logits).permute(0, 2, 1)
        # Gumbel Softmax (Hard)


        gumbels = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
        )  # ~Gumbel(0,1)
        gumbels = (logits + gumbels) / 0.5  # ~Gumbel(logits,tau)
        y_soft = gumbels.softmax(dim=1)
        index = y_soft.argsort(dim=1, descending=True)[:, :int(self.frequency_size*self.alpha), :]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(1, index, 1.0)
        mask = (y_hard - y_soft).detach() + y_soft
        # indices = y_soft.topk(int(self.frequency_size * (1-self.alpha)), dim=1, largest=False).indices
        # masked = y_soft.scatter(dim=1, index=indices, value=0.)
        # x_masked = (masked - y_soft).detach() + y_soft
        xf = xf.permute(0, 2, 1)

        x_inv = torch.fft.irfft(xf*mask, dim=1)
        x_var = x - x_inv

        if device == 'mps':
            x_var = x_var.to('mps')
            x_inv = x_inv.to('mps')
            entropy = (-y_soft * y_soft.log()).sum(dim=1).mean(dim=-1).to('mps')
        return x_var, x_inv,  entropy
    

class MLP(nn.Module):
    '''
    Multilayer perceptron to encode/decode high dimension representation of sequential data
    '''
    def __init__(self, 
                 f_in, 
                 f_out, 
                 hidden_dim=128, 
                 hidden_layers=2, 
                 dropout=0.05,
                 activation='tanh'): 
        super(MLP, self).__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise NotImplementedError

        layers = [nn.Linear(self.f_in, self.hidden_dim), 
                  self.activation, nn.Dropout(self.dropout)]
        for i in range(self.hidden_layers-2):
            layers += [nn.Linear(self.hidden_dim, self.hidden_dim),
                       self.activation, nn.Dropout(dropout)]
        
        layers += [nn.Linear(hidden_dim, f_out)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # x:     B x S x f_in
        # y:     B x S x f_out
        y = self.layers(x)
        return y
    

class KPLayer(nn.Module):
    """
    A demonstration of finding one step transition of linear system by DMD iteratively
    """
    def __init__(self): 
        super(KPLayer, self).__init__()
        
        self.K = None # B E E

    def one_step_forward(self, z, return_rec=False, return_K=False):
        B, input_len, E = z.shape
        assert input_len > 1, 'snapshots number should be larger than 1'
        x, y = z[:, :-1], z[:, 1:]

        # solve linear system
        if 'mps' in str(x.device):
            device = 'mps'
        else:
            device = None
        
        if device == 'mps':
            self.K = torch.linalg.lstsq(x.to('cpu'), y.to('cpu')).solution.to('mps')
        else:
            self.K = torch.linalg.lstsq(x, y).solution # B E E
    
        if torch.isnan(self.K).any():
            print('Encounter K with nan, replace K by identity matrix')
            self.K = torch.eye(self.K.shape[1]).to(self.K.device).unsqueeze(0).repeat(B, 1, 1)

        z_pred = torch.bmm(z[:, -1:], self.K)
        if return_rec:
            z_rec = torch.cat((z[:, :1], torch.bmm(x, self.K)), dim=1)
            return z_rec, z_pred

        return z_pred
    
    def forward(self, z, pred_len=1):
        assert pred_len >= 1, 'prediction length should not be less than 1'
        z_rec, z_pred= self.one_step_forward(z, return_rec=True)
        z_preds = [z_pred]
        for i in range(1, pred_len):
            z_pred = torch.bmm(z_pred, self.K)
            z_preds.append(z_pred)
        z_preds = torch.cat(z_preds, dim=1)
        return z_rec, z_preds


class KPLayerApprox(nn.Module):
    """
    Find koopman transition of linear system by DMD with multistep K approximation
    """
    def __init__(self): 
        super(KPLayerApprox, self).__init__()
        
        self.K = None # B E E
        self.K_step = None # B E E

    def forward(self, z, pred_len=1):
        # z:       B L E, koopman invariance space representation
        # z_rec:   B L E, reconstructed representation
        # z_pred:  B S E, forecasting representation
        B, input_len, E = z.shape
        assert input_len > 1, 'snapshots number should be larger than 1'
        x, y = z[:, :-1], z[:, 1:]

        # solve linear system
        self.K = torch.linalg.lstsq(x, y).solution # B E E

        if torch.isnan(self.K).any():
            print('Encounter K with nan, replace K by identity matrix')
            self.K = torch.eye(self.K.shape[1]).to(self.K.device).unsqueeze(0).repeat(B, 1, 1)

        z_rec = torch.cat((z[:, :1], torch.bmm(x, self.K)), dim=1) # B L E
        
        if pred_len <= input_len:
            self.K_step = torch.linalg.matrix_power(self.K, pred_len)
            if torch.isnan(self.K_step).any():
                print('Encounter multistep K with nan, replace it by identity matrix')
                self.K_step = torch.eye(self.K_step.shape[1]).to(self.K_step.device).unsqueeze(0).repeat(B, 1, 1)
            z_pred = torch.bmm(z[:, -pred_len:, :], self.K_step)
        else:
            self.K_step = torch.linalg.matrix_power(self.K, input_len)
            if torch.isnan(self.K_step).any():
                print('Encounter multistep K with nan, replace it by identity matrix')
                self.K_step = torch.eye(self.K_step.shape[1]).to(self.K_step.device).unsqueeze(0).repeat(B, 1, 1)
            temp_z_pred, all_pred = z, []
            for _ in range(math.ceil(pred_len / input_len)):
                temp_z_pred = torch.bmm(temp_z_pred, self.K_step)
                all_pred.append(temp_z_pred)
            z_pred = torch.cat(all_pred, dim=1)[:, :pred_len, :]

        return z_rec, z_pred
    

class TimeVarKP(nn.Module):
    """
    Koopman Predictor with DMD (analysitical solution of Koopman operator)
    Utilize local variations within individual sliding window to predict the future of time-variant term
    """
    def __init__(self,
                 enc_in=8,
                 input_len=96,
                 pred_len=96,
                 seg_len=24,
                 dynamic_dim=128,
                 encoder=None,
                 decoder=None,
                 multistep=False,
                ):
        super(TimeVarKP, self).__init__()
        self.input_len = input_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.seg_len = seg_len
        self.dynamic_dim = dynamic_dim
        self.multistep = multistep
        self.encoder, self.decoder = encoder, decoder            
        self.freq = math.ceil(self.input_len / self.seg_len)  # segment number of input
        self.step = math.ceil(self.pred_len / self.seg_len)   # segment number of output
        self.padding_len = self.seg_len * self.freq - self.input_len
        # Approximate mulitstep K by KPLayerApprox when pred_len is large
        self.dynamics = KPLayerApprox() if self.multistep else KPLayer() 

    def forward(self, x):
        # x: B L C
        B, L, C = x.shape

        res = torch.cat((x[:, L-self.padding_len:, :], x) ,dim=1)

        res = res.chunk(self.freq, dim=1)     # F x B P C, P means seg_len
        res = torch.stack(res, dim=1).reshape(B, self.freq, -1)   # B F PC

        res = self.encoder(res) # B F H
        x_rec, x_pred = self.dynamics(res, self.step) # B F H, B S H

        x_rec = self.decoder(x_rec) # B F PC
        x_rec = x_rec.reshape(B, self.freq, self.seg_len, self.enc_in)
        x_rec = x_rec.reshape(B, -1, self.enc_in)[:, :self.input_len, :]  # B L C
        
        x_pred = self.decoder(x_pred)     # B S PC
        x_pred = x_pred.reshape(B, self.step, self.seg_len, self.enc_in)
        x_pred = x_pred.reshape(B, -1, self.enc_in)[:, :self.pred_len, :] # B S C

        return x_rec, x_pred


class TimeInvKP(nn.Module):
    """
    Koopman Predictor with learnable Koopman operator
    Utilize lookback and forecast window snapshots to predict the future of time-invariant term
    """
    def __init__(self,
                 input_len=96,
                 pred_len=96,
                 dynamic_dim=128,
                 encoder=None,
                 decoder=None):
        super(TimeInvKP, self).__init__()
        self.dynamic_dim = dynamic_dim
        self.input_len = input_len
        self.pred_len = pred_len
        self.encoder = encoder
        self.decoder = decoder

        K_init = torch.randn(self.dynamic_dim, self.dynamic_dim)
        U, _, V = torch.svd(K_init) # stable initialization
        self.K = nn.Linear(self.dynamic_dim, self.dynamic_dim, bias=False)
        self.K.weight.data = torch.mm(U, V.t())
    
    def forward(self, x):
        # x: B L C
        res = x.transpose(1, 2) # B C L
        res = self.encoder(res) # B C H
        res = self.K(res) # B C H
        res = self.decoder(res) # B C S
        res = res.transpose(1, 2) # B S C

        return res


class Model(nn.Module):
    '''
    Koopman Forecasting Model
    '''
    def __init__(self, configs):
        super(Model, self).__init__()
        self.mask_spectrum = configs.mask_spectrum
        self.enc_in = configs.enc_in
        self.input_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.seg_len = configs.seg_len
        self.num_blocks = configs.num_blocks
        self.dynamic_dim = configs.dynamic_dim
        self.hidden_dim = configs.hidden_dim
        self.hidden_layers = configs.hidden_layers
        self.multistep = configs.multistep
        self.fourier = configs.fourier

        if self.fourier == 'filter':
            self.disentanglement = FourierFilter(self.mask_spectrum)
        elif self.fourier == 'select':
            self.disentanglement = FourierSelector(self.input_len, configs.alpha)
        
        # shared encoder/decoder to make koopman embedding consistent
        self.time_inv_encoder = MLP(f_in=self.input_len, f_out=self.dynamic_dim, activation='relu',
                    hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers)
        self.time_inv_decoder = MLP(f_in=self.dynamic_dim, f_out=self.pred_len, activation='relu',
                           hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers)
        self.time_inv_kps = self.time_var_kps = nn.ModuleList([
                                TimeInvKP(input_len=self.input_len,
                                    pred_len=self.pred_len, 
                                    dynamic_dim=self.dynamic_dim,
                                    encoder=self.time_inv_encoder, 
                                    decoder=self.time_inv_decoder)
                                for _ in range(self.num_blocks)])

        # shared encoder/decoder to make koopman embedding consistent
        self.time_var_encoder = MLP(f_in=self.seg_len*self.enc_in, f_out=self.dynamic_dim, activation='tanh',
                           hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers)
        self.time_var_decoder = MLP(f_in=self.dynamic_dim, f_out=self.seg_len*self.enc_in, activation='tanh',
                           hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers)
        self.time_var_kps = nn.ModuleList([
                    TimeVarKP(enc_in=configs.enc_in,
                        input_len=self.input_len,
                        pred_len=self.pred_len,
                        seg_len=self.seg_len,
                        dynamic_dim=self.dynamic_dim,
                        encoder=self.time_var_encoder,
                        decoder=self.time_var_decoder,
                        multistep=self.multistep)
                    for _ in range(self.num_blocks)])
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc: B L C

        # Series Stationarization adopted from NSformer
        mean_enc = x_enc.mean(1, keepdim=True).detach() # B x 1 x E
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / std_enc

        # Koopman Forecasting
        residual, forecast = x_enc, None
        for i in range(self.num_blocks):
            if self.fourier == 'filter':
                time_var_input, time_inv_input = self.disentanglement(residual)
            elif self.fourier == 'select':
                time_var_input, time_inv_input, fourier_dist = self.disentanglement(residual)
            time_inv_output = self.time_inv_kps[i](time_inv_input)
            time_var_backcast, time_var_output = self.time_var_kps[i](time_var_input)
            residual = residual - time_var_backcast
            if forecast is None:
                forecast = (time_inv_output + time_var_output)
            else:
                forecast += (time_inv_output + time_var_output)

        # Series Stationarization adopted from NSformer
        if self.fourier == 'filter':
            res = forecast * std_enc + mean_enc
        elif self.fourier == 'select':
            res = forecast * std_enc + mean_enc, fourier_dist
        return res
