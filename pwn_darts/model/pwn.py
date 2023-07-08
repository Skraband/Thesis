from .spectral_rnn.spectral_rnn import SpectralRNN, SpectralRNNConfig
from .spectral_rnn.manifold_optimization import ManifoldOptimizer
from .spectral_rnn.cgRNN import clip_grad_value_complex_
from .transformer import TransformerPredictor, TransformerConfig
from .maf import MAFEstimator
from .cwspn import CWSPN, CWSPNConfig
from .model import Model

import numpy as np
import torch
import torch.nn as nn

from darts.darts_cnn.model_search import Network as CWSPNModelSearch
from darts.darts_rnn.model_search import RNNModelSearch
from datetime import datetime
# Use GPU if avaiable
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_arch_params(srnn=True):

    if srnn:
        arch_params_raw = np.array([[0.2044706791639328, 0.23126950860023499, 0.20898419618606567, 0.11867085844278336, 0.236604705452919], [0.20995227992534637, 0.2200889140367508, 0.21184001863002777, 0.13608209788799286, 0.222036674618721], [0.22023913264274597, 0.21395696699619293, 0.21425333619117737, 0.13705332577228546, 0.21449723839759827], [0.20980963110923767, 0.2122781127691269, 0.21050076186656952, 0.15452083945274353, 0.21289068460464478], [0.21522517502307892, 0.20951567590236664, 0.20959848165512085, 0.15596798062324524, 0.20969270169734955], [0.22860990464687347, 0.20411226153373718, 0.20424720644950867, 0.1586771458387375, 0.20435355603694916], [0.20649385452270508, 0.20787662267684937, 0.2067175954580307, 0.1708189696073532, 0.20809291303157806], [0.20908930897712708, 0.20630207657814026, 0.20631638169288635, 0.1719292253255844, 0.2063630074262619], [0.21535076200962067, 0.20343899726867676, 0.2034461498260498, 0.174265056848526, 0.20349901914596558], [0.2309189885854721, 0.19737671315670013, 0.19673915207386017, 0.17816618084907532, 0.19679902493953705], [0.20405100286006927, 0.20484332740306854, 0.20409861207008362, 0.1821068823337555, 0.20490020513534546], [0.20546157658100128, 0.20392201840877533, 0.20391331613063812, 0.18276670575141907, 0.20393632352352142], [0.2087959498167038, 0.2023080736398697, 0.20227552950382233, 0.18432138860225677, 0.20229901373386383], [0.21620698273181915, 0.19916710257530212, 0.19871705770492554, 0.18716329336166382, 0.19874553382396698], [0.22971001267433167, 0.19518937170505524, 0.19209490716457367, 0.19087497889995575, 0.19213077425956726], [0.20234623551368713, 0.20348253846168518, 0.20235981047153473, 0.18831117451190948, 0.20350025594234467], [0.20305168628692627, 0.20277611911296844, 0.2027752846479416, 0.18861447274684906, 0.20278240740299225], [0.20506058633327484, 0.20176084339618683, 0.2017555981874466, 0.18966037034988403, 0.2017626315355301], [0.20930473506450653, 0.1998327672481537, 0.19959966838359833, 0.19165492057800293, 0.19960789382457733], [0.21641166508197784, 0.1975902020931244, 0.195729598402977, 0.19452892243862152, 0.19573956727981567], [0.22850169241428375, 0.1967296004295349, 0.1883469671010971, 0.1980632245540619, 0.18835853040218353], [0.20199459791183472, 0.20198951661586761, 0.20199789106845856, 0.1920265555381775, 0.20199143886566162], [0.20262153446674347, 0.20166967809200287, 0.20166590809822083, 0.19237329065799713, 0.2016695737838745], [0.20395700633525848, 0.20099537074565887, 0.20097561180591583, 0.19309405982494354, 0.2009778916835785], [0.20675043761730194, 0.1997351050376892, 0.19950439035892487, 0.19450289011001587, 0.19950717687606812], [0.21122466027736664, 0.19840364158153534, 0.19688479602336884, 0.1965988278388977, 0.1968880444765091], [0.2183569371700287, 0.19831743836402893, 0.19198070466518402, 0.1993577927350998, 0.19198714196681976], [0.2280554175376892, 0.2007911652326584, 0.18421506881713867, 0.2027120590209961, 0.18422622978687286], [0.2014298141002655, 0.2014985829591751, 0.20142699778079987, 0.19414475560188293, 0.2014998495578766], [0.2018612027168274, 0.20125219225883484, 0.20125235617160797, 0.19438126683235168, 0.20125296711921692], [0.20282091200351715, 0.20076464116573334, 0.20075105130672455, 0.19491244852542877, 0.20075096189975739], [0.20480796694755554, 0.19984997808933258, 0.19968612492084503, 0.19596940279006958, 0.19968651235103607], [0.20793665945529938, 0.19890327751636505, 0.1977963149547577, 0.19756709039211273, 0.19779665768146515], [0.21273162961006165, 0.19896261394023895, 0.19429175555706024, 0.19972163438796997, 0.194292351603508], [0.21897579729557037, 0.20106372237205505, 0.1887521892786026, 0.20245850086212158, 0.18874983489513397], [0.22957831621170044, 0.20659056305885315, 0.17823675274848938, 0.2073718011379242, 0.17822252213954926]])
        arch_params = (arch_params_raw == arch_params_raw.max(axis=1, keepdims=1)).astype(float)
        arch_params_torch = torch.tensor(arch_params).cuda()

    else:
        arch_params_raw = [[0.19716767966747284, 0.12434124946594238, 0.14479579031467438, 0.16580961644649506, 0.1916297823190689, 0.17625582218170166], [0.19316959381103516, 0.12926524877548218, 0.13932734727859497, 0.20224907994270325, 0.1219061091542244, 0.21408262848854065], [0.17145590484142303, 0.13171736896038055, 0.18166901171207428, 0.19344878196716309, 0.15573011338710785, 0.16597887873649597], [0.1940115988254547, 0.1294841170310974, 0.19451498985290527, 0.19662517309188843, 0.13976766169071198, 0.1455964744091034], [0.17357401549816132, 0.1343383491039276, 0.18129891157150269, 0.20344628393650055, 0.15183396637439728, 0.15550850331783295], [0.17802764475345612, 0.1410401463508606, 0.17493470013141632, 0.1858380138874054, 0.15979167819023132, 0.16036781668663025], [0.1816277652978897, 0.12539248168468475, 0.15040381252765656, 0.19027550518512726, 0.1806478351354599, 0.17165258526802063], [0.1797628402709961, 0.13560672104358673, 0.19025321304798126, 0.19118529558181763, 0.15743204951286316, 0.14575985074043274], [0.17241984605789185, 0.1406448632478714, 0.17019282281398773, 0.18222084641456604, 0.17612969875335693, 0.15839192271232605], [0.1784159243106842, 0.13855640590190887, 0.17900541424751282, 0.18851955235004425, 0.1726435422897339, 0.14285917580127716], [0.1666666716337204, 0.1666666716337204, 0.1666666716337204, 0.1666666716337204, 0.1666666716337204, 0.1666666716337204], [0.1666666716337204, 0.1666666716337204, 0.1666666716337204, 0.1666666716337204, 0.1666666716337204, 0.1666666716337204], [0.1666666716337204, 0.1666666716337204, 0.1666666716337204, 0.1666666716337204, 0.1666666716337204, 0.1666666716337204], [0.1666666716337204, 0.1666666716337204, 0.1666666716337204, 0.1666666716337204, 0.1666666716337204, 0.1666666716337204]]
        arch_params = (arch_params_raw == arch_params_raw.max(axis=1, keepdims=1)).astype(float)
        arch_params_torch = torch.tensor(arch_params).cuda()

    return arch_params_torch

class PWN(Model):

    def __init__(self, s_config: SpectralRNNConfig, c_config: CWSPNConfig, train_spn_on_gt=True,
                 train_spn_on_prediction=False, train_rnn_w_ll=False, weight_mse_by_ll=None, always_detach=False,
                 westimator_early_stopping=5, step_increase=False, westimator_stop_threshold=.5,
                 westimator_final_learn=2, ll_weight=0.5, ll_weight_inc_dur=20, use_transformer=False, use_maf=False,
                 smape_target=False):

        assert train_spn_on_gt or train_spn_on_prediction
        assert not train_rnn_w_ll or train_spn_on_gt

        self.srnn = SpectralRNN(s_config) if not use_transformer else TransformerPredictor(
            TransformerConfig(normalize_fft=True, window_size=s_config.window_size,
                              fft_compression=s_config.fft_compression))
        self.westimator = CWSPN(c_config) if not use_maf else MAFEstimator()

        self.train_spn_on_gt = train_spn_on_gt
        self.train_spn_on_prediction = train_spn_on_prediction
        self.train_rnn_w_ll = train_rnn_w_ll
        self.weight_mse_by_ll = weight_mse_by_ll
        self.always_detach = always_detach

        self.westimator_early_stopping = westimator_early_stopping
        self.westimator_stop_threshold = westimator_stop_threshold
        self.westimator_final_learn = westimator_final_learn
        self.ll_weight = ll_weight
        self.ll_weight_inc_dur = ll_weight_inc_dur
        self.step_increase = step_increase
        self.use_transformer = use_transformer
        self.use_maf = use_maf
        self.smape_target = smape_target

    def train(self, x_in, y_in, val_x, val_y, embedding_sizes, batch_size=256, epochs=70, lr=0.004, lr_decay=0.97):

        # TODO: Adjustment for complex optimization, needs documentation
        if type(self.srnn) == TransformerPredictor:
            lr /= 10
        elif self.srnn.config.rnn_layer_config.use_cg_cell:
            lr /= 4

        # Model is trained jointly on all groups
        x_ = np.concatenate(list(x_in.values()), axis=0)
        y_ = np.concatenate(list(y_in.values()), axis=0)[:, :, -1]

        if self.srnn.final_amt_pred_samples is None:
            self.srnn.final_amt_pred_samples = next(iter(y_in.values())).shape[1]

        # Init SRNN
        x = torch.from_numpy(x_).float().to(device)
        y = torch.from_numpy(y_).float().to(device)
        self.srnn.config.embedding_sizes = embedding_sizes
        self.srnn.build_net()

        self.westimator.stft_module = self.srnn.net.stft

        if self.srnn.config.use_searched_srnn:
            search_stft = self.srnn.net.stft
            emsize = 300
            nhid = 300
            nhidlast = 300
            ntokens = 10000
            dropout = 0
            dropouth = 0
            dropouti = 0
            dropoute = 0
            dropoutx = 0
            config_layer = self.srnn.config
            srnn_search = RNNModelSearch(search_stft, config_layer, ntokens, emsize, nhid, nhidlast,
                                     dropout, dropouth, dropoutx, dropouti, dropoute).cuda()
            # set arch weights according to searched cell
            arch_params = load_arch_params(True)
            srnn_search.fix_arch_params(arch_params)
            # set searched srnn arch as srnn net
            self.srnn.net = srnn_search
        # Init westimator
        westimator_x_prototype, westimator_y_prototype = self.westimator.prepare_input(x[:1, :, -1], y[:1])
        self.westimator.input_sizes = westimator_x_prototype.shape[1], westimator_y_prototype.shape[1]
        self.westimator.create_net()

        if self.westimator.config.use_searched_cwspn:
            in_seq_length = self.westimator.input_sizes[0] * (
                2 if self.westimator.use_stft else 1)  # input sequence length into the WeightNN
            output_length = self.westimator.num_sum_params + self.westimator.num_leaf_params  # combined length of sum and leaf params
            sum_params = self.westimator.num_sum_params
            cwspn_weight_nn_search = CWSPNModelSearch(in_seq_length, output_length, sum_params, layers=1, steps=4).cuda()
            self.westimator.weight_nn = cwspn_weight_nn_search


        prediction_loss = lambda error: error.mean()
        ll_loss = lambda out: -1 * torch.logsumexp(out, dim=1).mean()
        if self.smape_target:
            smape_adjust = 2  # Move all values into the positive space
            p_base_loss = lambda out, label: 2 * (torch.abs(out - label) /
                                                  (torch.abs(out + smape_adjust) +
                                                   torch.abs(label + smape_adjust))).mean(axis=1)
        # MSE target
        else:
            p_base_loss = lambda out, label: nn.MSELoss(reduction='none')(out, label).mean(axis=1)

        srnn_parameters = list(self.srnn.net.parameters())
        westimator_parameters = self.westimator.parameters()

        amt_param = sum([p.numel() for p in self.srnn.net.parameters()])
        amt_param_w = sum([p.numel() for p in self.westimator.parameters()])

        srnn_optimizer = ManifoldOptimizer(srnn_parameters, lr, torch.optim.RMSprop, alpha=0.9) \
            if type(self.srnn.config) == SpectralRNNConfig and self.srnn.config.rnn_layer_config.use_cg_cell \
            else torch.optim.RMSprop(srnn_parameters, lr=lr, alpha=0.9)
        westimator_optimizer = torch.optim.Adam(westimator_parameters, lr=1e-4)

        if self.train_rnn_w_ll:
            current_ll_weight = 0
            ll_weight_history = []
            ll_weight_increase = self.ll_weight / self.ll_weight_inc_dur
        elif self.train_spn_on_prediction:
            def ll_loss_pred(out, error):
                return (-1 * torch.logsumexp(out, dim=1) * (error ** -2)).mean() * 1e-4

        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=srnn_optimizer, gamma=lr_decay)

        westimator_losses = []
        srnn_losses = []
        srnn_losses_p = []
        srnn_losses_ll = []

        stop_cspn_training = False
        westimator_patience_counter = 0
        westimator_losses_epoch = []

        self.srnn.net.train()

        if hasattr(self.westimator, 'spn'):
            self.westimator.spn.train()
            self.westimator.weight_nn.train()
        else:
            self.westimator.model.train()

        if self.srnn.config.use_cached_predictions:
            import pickle
            with open('srnn_train.pkl', 'rb') as f:
                all_predictions, all_f_cs = pickle.load(f)

                all_predictions = torch.from_numpy(all_predictions).to(device)
                all_f_cs = torch.from_numpy(all_f_cs).to(device)

        val_errors = []
        print(f'Starting Training of {self.identifier} model')
        for epoch in range(epochs):
            idx_batches = torch.randperm(x.shape[0], device=device).split(batch_size)

            #if epoch % 1000:
            #    model_base_path = 'res/models/'
            #    model_name = f'{self.identifier}-{str(epoch)}'
            #    model_filepath = f'{model_base_path}{model_name}'
            #    self.save(model_filepath)

            srnn_loss_p_e = 0
            srnn_loss_ll_e = 0

            pi = torch.tensor(np.pi)
            srnn_loss_e = 0
            westimator_loss_e = 0
            for i, idx in enumerate(idx_batches):
                # 1000th save every epoch
                #if (i+1) % 1 == 0:
                #    model_base_path = 'res/models/'
                #    model_name = f'{self.identifier}'
                #    experiment_id = f'{datetime.now().strftime("%m_%d_%Y_%H_%M_%S")}__{model_name}'
                #    model_filepath = f'{model_base_path}{experiment_id}'
                #    self.save(model_filepath)

                batch_x, batch_y = x[idx].detach().clone(), y[idx, :].detach().clone()
                batch_westimator_x, batch_westimator_y = self.westimator.prepare_input(batch_x[:, :, -1], batch_y)

                if self.srnn.config.use_cached_predictions:
                    batch_p = all_predictions.detach().clone()[idx]
                    batch_fc = all_f_cs.detach().clone()[idx]

                if self.train_spn_on_gt:
                    westimator_optimizer.zero_grad()
                    if not stop_cspn_training or epoch >= epochs - self.westimator_final_learn:
                        out_w, _ = self.call_westimator(batch_westimator_x, batch_westimator_y)

                        if hasattr(self.westimator, 'spn'):
                            gt_ll = out_w
                            westimator_loss = ll_loss(gt_ll)
                            westimator_loss.backward()
                            westimator_optimizer.step()
                        else:
                            if self.westimator.use_made:
                                raise NotImplementedError  # MADE not implemented here

                            u, log_det = out_w

                            negloglik_loss = 0.5 * (u ** 2).sum(dim=1)
                            negloglik_loss += 0.5 * self.westimator.final_input_sizes * np.log(2 * pi)
                            negloglik_loss -= log_det
                            negloglik_loss = torch.mean(negloglik_loss)

                            negloglik_loss.backward()
                            westimator_loss = negloglik_loss.item()
                            westimator_optimizer.step()
                            westimator_optimizer.zero_grad()

                    else:
                        westimator_loss = westimator_losses_epoch[-1]

                    westimator_loss_e += westimator_loss.detach()

                # Also zero grads for westimator, s.t. old grads dont influence the optimization
                srnn_optimizer.zero_grad()
                westimator_optimizer.zero_grad()

                if not self.srnn.config.use_cached_predictions:
                    if self.srnn.config.use_searched_srnn:
                        prediction, f_c = self.srnn.net(batch_x, batch_y, self.srnn.net.weights,
                                                        return_coefficients=True)
                    else:
                        prediction, f_c = self.srnn.net(batch_x, batch_y, return_coefficients=True)
                else:
                    prediction, f_c = batch_p, batch_fc

                if self.westimator.use_stft:
                    prediction_ll, w_in = self.call_westimator(batch_westimator_x, f_c.reshape((f_c.shape[0], -1))
                                        if self.train_rnn_w_ll and not self.always_detach else f_c.reshape((f_c.shape[0], -1)).detach())
                else:
                    prediction_ll, w_in = self.call_westimator(batch_westimator_x, prediction
                                        if self.train_rnn_w_ll and not self.always_detach else prediction.detach())

                #if type(self.srnn) == TransformerPredictor:
                #    num_to_cut = int(prediction_raw.shape[1]/3)
                #    prediction = prediction_raw[:,num_to_cut:]
                #else:
                #    prediction = prediction_raw

                error = p_base_loss(prediction, batch_y)
                p_loss = prediction_loss(error)

                if self.train_rnn_w_ll:
                    l_loss = ll_loss(prediction_ll)

                    if self.weight_mse_by_ll is None:
                        srnn_loss = (1 - current_ll_weight) * p_loss + current_ll_weight * l_loss
                    else:
                        local_ll = torch.logsumexp(prediction_ll, dim=1)
                        local_ll = local_ll - local_ll.max()  # From 0 to -inf
                        local_ll = local_ll / local_ll.min()  # From 0 to 1 -> low LL is 1, high LL is 0: Inverse Het
                        local_ll = local_ll / local_ll.mean()  # Scale it to mean = 1

                        if self.weight_mse_by_ll == 'het':
                            # Het: low LL is 0, high LL is 1
                            local_ll = local_ll.max() - local_ll

                        srnn_loss = p_loss * (self.ll_weight - current_ll_weight) + \
                                    current_ll_weight * (error * local_ll).mean()
                else:
                    srnn_loss = p_loss
                    l_loss = 0

                if not self.srnn.config.use_cached_predictions:
                    srnn_loss.backward()

                    if self.srnn.config.clip_gradient_value > 0:
                        clip_grad_value_complex_(srnn_parameters, self.srnn.config.clip_gradient_value)

                    srnn_optimizer.step()

                if self.train_spn_on_prediction:
                    if hasattr(self.westimator, 'spn'):
                        westimator_loss = ll_loss_pred(prediction_ll, error.detach())

                        westimator_loss.backward()
                        westimator_optimizer.step()
                    else:
                        if type(prediction_ll) == tuple:
                            u, log_det = prediction_ll

                            negloglik_loss = 0.5 * (u ** 2).sum(dim=1)
                            negloglik_loss += 0.5 * self.westimator.final_input_sizes * np.log(2 * pi)
                            negloglik_loss -= log_det
                            negloglik_loss = torch.mean(negloglik_loss * (error ** -2)) * 1e-4

                        else:
                            mu, logp = torch.chunk(prediction_ll, 2, dim=1)
                            u = (w_in - mu) * torch.exp(0.5 * logp)

                            negloglik_loss = 0.5 * (u ** 2).sum(dim=1)
                            negloglik_loss += 0.5 * w_in.shape[1] * np.log(2 * pi)
                            negloglik_loss -= 0.5 * torch.sum(logp, dim=1)

                            negloglik_loss = torch.mean(negloglik_loss)

                        negloglik_loss.backward()
                        westimator_loss = negloglik_loss.item()
                        westimator_optimizer.step()

                    westimator_loss_e += westimator_loss.detach()

                l_loss = l_loss.detach() if not type(l_loss) == int else l_loss
                srnn_loss_p_e += p_loss.item()
                srnn_loss_ll_e += l_loss
                srnn_loss_e += srnn_loss.item()

                westimator_losses.append(westimator_loss.detach().cpu().numpy())
                srnn_losses.append(srnn_loss.detach().cpu().numpy())
                srnn_losses_p.append(p_loss.detach().cpu().numpy())
                srnn_losses_ll.append(l_loss)

                if i == 0:
                #if (i + 1) % 10 == 0:
                    print(f'Epoch {epoch + 1} / {epochs}: Step {(i + 1)} / {len(idx_batches)}. '
                          f'Avg. WCSPN Loss: {westimator_loss_e / (i + 1)} '
                          f'Avg. SRNN Loss: {srnn_loss_e / (i + 1)}')

            lr_scheduler.step()

            if epoch < self.ll_weight_inc_dur and self.train_rnn_w_ll:
                if self.step_increase:
                    current_ll_weight = 0
                else:
                    current_ll_weight += ll_weight_increase
            elif self.train_rnn_w_ll:
                current_ll_weight = self.ll_weight

            westimator_loss_epoch = westimator_loss_e / len(idx_batches)
            srnn_loss_epoch = srnn_loss_e / len(idx_batches)
            print(f'Epoch {epoch + 1} / {epochs} done.'
                  f'Avg. WCSPN Loss: {westimator_loss_epoch} '
                  f'Avg. SRNN Loss: {srnn_loss_epoch}')

            print(f'Avg. SRNN-Prediction-Loss: {srnn_loss_p_e / len(idx_batches)}')
            print(f'Avg. SRNN-LL-Loss: {srnn_loss_ll_e / len(idx_batches)}')

            if len(westimator_losses_epoch) > 0 and not stop_cspn_training and \
                    not westimator_loss_epoch < westimator_losses_epoch[-1] - self.westimator_stop_threshold and not \
                    self.train_spn_on_prediction:
                westimator_patience_counter += 1

                print(f'Increasing patience counter to {westimator_patience_counter}')

                if westimator_patience_counter >= self.westimator_early_stopping:
                    stop_cspn_training = True
                    print('WCSPN training stopped!')

            else:
                westimator_patience_counter = 0

            westimator_losses_epoch.append(westimator_loss_epoch)

            if False and epoch % 3 == 0:
                pred_val, _ = self.predict({key: x for key, x in val_x.items() if len(x) > 0}, mpe=False)
                self.srnn.net.train()
                self.westimator.spn.train()
                self.westimator.weight_nn.train()

                val_mse = np.mean([((p - val_y[key][:, :, -1]) ** 2).mean() for key, p in pred_val.items()])
                val_errors.append(val_mse)

        if not self.srnn.config.use_cached_predictions:
            predictions = []
            f_cs = []

            with torch.no_grad():
                print(x.shape[0] // batch_size + 1)
                for i in range(x.shape[0] // batch_size + 1):
                    batch_x, batch_y = x[i * batch_size:(i + 1) * batch_size], \
                                       y[i * batch_size:(i + 1) * batch_size]
                    if self.srnn.config.use_searched_srnn:
                        prediction, f_c = self.srnn.net(batch_x, batch_y, self.srnn.net.weights,
                                                            return_coefficients=True)
                    else:
                        prediction, f_c = self.srnn.net(batch_x, batch_y, return_coefficients=True)

                    predictions.append(prediction)
                    f_cs.append(f_c)

            predictions = torch.cat(predictions, dim=0).detach().cpu().numpy()
            f_cs = torch.cat(f_cs, dim=0).detach().cpu().numpy()

            import pickle
            with open('srnn_train.pkl', 'wb') as f:
                pickle.dump((predictions, f_cs), f)

        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.size': 48, 'figure.figsize': (60, 40)})
        index = list(range(len(westimator_losses)))
        plt.ylabel('LL')
        plt.plot(index, westimator_losses, label='WCSPN-Loss (Negative LL)', color='blue')
        plt.plot(index, srnn_losses_ll, label='SRNN-Loss (Negative LL)', color='green')
        plt.legend(loc='upper right')

        ax2 = plt.twinx()
        ax2.set_ylabel('MSE', color='red')
        ax2.plot(index, srnn_losses, label='SRNN-Loss Total', color='magenta')
        ax2.plot(index, srnn_losses_p, label='SRNN-Loss Prediction', color='red')
        ax2.legend(loc='upper left')

        plt.savefig('res/plots/0_PWN_Training_losses')

        plt.clf()
        plt.plot(val_errors)
        plt.savefig('res/plots/0_PWN_Val_MSE')
        print(val_errors)

        if self.train_rnn_w_ll:
            plt.clf()
            plt.plot(ll_weight_history)
            plt.ylabel('SRNN LL-Loss Weight (percentage of total loss)')
            plt.title('LL Weight Warmup')
            plt.savefig('res/plots/0_PWN_LLWeightWarmup')

    @torch.no_grad()
    def predict(self, x, batch_size=1024, pred_label='', mpe=False):
        predictions, f_c = self.srnn.predict(x, batch_size, pred_label=pred_label, return_coefficients=True)

        x_ = {key: x_[:, :, -1] for key, x_ in x.items()}

        f_c_ = {key: f.reshape((f.shape[0], -1)) for key, f in f_c.items()}

        if self.westimator.use_stft:
            ll = self.westimator.predict(x_, f_c_, stft_y=False, batch_size=batch_size)
        else:
            ll = self.westimator.predict(x_, predictions, batch_size=batch_size)

        if mpe:
            y_empty = {key: np.zeros((x[key].shape[0], self.srnn.net.amt_prediction_samples)) for key in x.keys()}
            predictions_mpe = self.westimator.predict_mpe({key: x_.copy() for key, x_ in x.items()},
                                                           y_empty, batch_size=batch_size)
            lls_mpe = self.westimator.predict(x_, {key: v[0] for key, v in predictions_mpe.items()},
                                              stft_y=False, batch_size=batch_size)

            # predictions, likelihoods, likelihoods_mpe, predictions_mpe
            return predictions, ll, lls_mpe, {key: v[1] for key, v in predictions_mpe.items()}

        else:
            return predictions, ll

    def save(self, filepath):
        self.srnn.save(filepath)
        self.westimator.save(filepath)

    def load(self, filepath):
        self.srnn.load(filepath)
        self.westimator.load(filepath)
        self.westimator.stft_module = self.srnn.net.stft

    def call_westimator(self, x, y):

        y_ = torch.stack([y.real, y.imag], dim=-1) if torch.is_complex(y) else y

        if hasattr(self.westimator, 'spn'):
            sum_params, leaf_params = self.westimator.weight_nn(x.reshape((x.shape[0], x.shape[1] *
                                                                           (2 if self.westimator.use_stft else 1),)))
            self.westimator.args.param_provider.sum_params = sum_params
            self.westimator.args.param_provider.leaf_params = leaf_params

            return self.westimator.spn(y_), y_
        else:
            val_in = torch.cat([x, y_], dim=1).reshape((x.shape[0], -1))
            return self.westimator.model(val_in), val_in
