import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import os
import time
import math

import utils.util as utils
from data import create_dataLoader
from models import create_model
from models.self_distillation import SelfDistillationModel
from models.fusion_module import FusionModule


class Trainer():

    def __init__(self, opt, logger):

        self.opt = opt
        self.opt.n_epochs = 90
        self.opt.lr_decay_iters = [30, 60, 80]
        self.opt.train_batch_size = 256
        self.opt.test_batch_size = 256
        self.opt.isTrain = True
        self.logger = logger
        self.device = torch.device(f'cuda:{opt.gpu_ids[0]}') if torch.cuda.is_available() else 'cpu'

        self.epochs = opt.n_epochs
        self.start_epochs = opt.start_epoch
        self.train_batch_size = self.opt.train_batch_size
        self.temperature = self.opt.temperature

        dataLoader = create_dataLoader(opt)
        self.trainLoader = dataLoader.trainLoader
        self.testLoader = dataLoader.testLoader

        self.criterion_CE = nn.CrossEntropyLoss().to(self.device)
        self.criterion_KL = nn.KLDivLoss(reduction='batchmean').to(self.device)

        self.model_num = opt.model_num
        self.models = []
        self.optimizers = []
        for i in range(self.model_num):
            model = create_model(opt).to(self.device)
            optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum,
                                  weight_decay=opt.weight_decay,
                                  nesterov=True)
            self.models.append(model)
            self.optimizers.append(optimizer)

        self.init_self_ditsllation_models()
        self.init_fusion_module()

        self.leader_model = create_model(self.opt, leader=True, trans_fusion_info=(self.fusion_channel, self.model_num)).to(self.device)
        self.leader_optimizer = optim.SGD(self.leader_model.parameters(), lr=opt.lr, momentum=opt.momentum,
                                          weight_decay=self.opt.leader_weight_decay,
                                          nesterov=True)

    def init_self_ditsllation_models(self):

        input_size = (1, 3, 224, 224)

        noise_input = torch.randn(input_size).to(self.device)
        self.models[0](noise_input)
        trans_input = list(self.models[0].total_feature_maps.values())[-1]
        self.fusion_channel = trans_input.size(1)
        self.fusion_spatil = trans_input.size(2)

        self.sd_models = []
        self.sd_optimizers = []
        self.sd_schedulers = []
        for i in range(1, self.model_num):
            sd_model = SelfDistillationModel(input_channel=trans_input.size(1),
                                             layer_num=len(self.models[0].extract_layers) - 1).to(self.device)
            sd_optimizer = optim.Adam(sd_model.parameters(), weight_decay=self.opt.weight_decay)
            sd_scheduler = utils.get_scheduler(sd_optimizer, self.opt)
            self.sd_models.append(sd_model)
            self.sd_optimizers.append(sd_optimizer)
            self.sd_schedulers.append(sd_scheduler)

        self.sd_leader_model = SelfDistillationModel(input_channel=trans_input.size(1),
                                                     layer_num=len(self.models[0].extract_layers) - 1).to(self.device)
        self.sd_leader_optimizer = optim.Adam(self.sd_leader_model.parameters(), weight_decay=self.opt.weight_decay)
        self.sd_leader_scheduler = utils.get_scheduler(self.sd_leader_optimizer, self.opt)

    def init_fusion_module(self):

        self.num_classes = 1000

        self.fusion_module = FusionModule(self.fusion_channel, self.num_classes,
                                          self.fusion_spatil, model_num=self.model_num).to(self.device)
        self.fusion_optimizer = optim.SGD(self.fusion_module.parameters(), lr=self.opt.lr, momentum=self.opt.momentum,
                                          weight_decay=1e-5,
                                          nesterov=True)

    def train(self):

        if self.opt.dataset == 'imagenet':
            topk = (1, 5)
        else:
            topk = (1,)

        best_acc = [0.0] * self.model_num
        best_epoch = [1] * self.model_num
        best_avg_acc = 0.0
        best_ens_acc = 0.0
        best_avg_epoch = 1
        best_ens_epoch = 1
        best_fusion_acc = 0.0
        best_fusion_epoch = 1
        best_leader_acc = 0.0
        best_leader_epoch = 1

        for epoch in range(self.start_epochs, self.epochs):


            self.lambda_warmup(epoch)
            self.train_with_test(epoch, topk=topk)

            test_losses, test_acc, test_top5_acc, test_avg_acc, test_ens_acc = self.test(epoch, topk=topk)

            for i in range(self.model_num):
                self.save_models(self.models[i], epoch, str(i), self.opt, isbest=False)
                if test_acc[i].avg > best_acc[i]:
                    best_acc[i] = test_acc[i].avg
                    best_epoch[i] = epoch
                    self.save_models(self.models[i], epoch, str(i), self.opt, isbest=True)

            if test_acc[-2].avg > best_fusion_acc:
                self.save_models(self.fusion_module, epoch, 'fusion', self.opt, isbest=True)
                best_fusion_acc = test_acc[-2].avg
                best_fusion_epoch = epoch

            if test_acc[-1].avg > best_leader_acc:
                self.save_models(self.leader_model, epoch, 'leader', self.opt, isbest=True)
                best_leader_acc = test_acc[-1].avg
                best_leader_epoch = epoch

            if test_avg_acc.avg > best_avg_acc:
                best_avg_acc = test_avg_acc.avg
                best_avg_epoch = epoch
            if test_ens_acc.avg > best_ens_acc:
                best_ens_acc = test_ens_acc.avg
                best_ens_epoch = epoch

            for scheduler in self.sd_schedulers:
                scheduler.step()
            self.sd_leader_scheduler.step()

        best_msg = 'Best Models: '
        self.logger.info(
            'Best Average/Ensemble Epoch{}:{:.2f}/Epoch{}:{:.2f}'.format(best_avg_epoch, float(best_avg_acc),
                                                                         best_ens_epoch, float(best_ens_acc)))
        for i in range(self.model_num):
            best_msg += 'Epoch {}:{:.2f}/'.format(best_epoch[i], float(best_acc[i]))
        self.logger.info(
            'Model[Fusion]/[Leader] Epoch{}:{:.2f}/Epoch{}:{:.2f}'.format(best_fusion_epoch, float(best_fusion_acc),
                                                                          best_leader_epoch, float(best_leader_acc)))
        self.logger.info(best_msg)

    def train_with_test(self, epoch, topk=(1,)):

        accuracy = []
        losses = []
        ce_losses = []
        dml_losses = []
        diversity_losses = []
        self_distillation_feature_losses = []
        self_distillation_attention_losses = []
        self_distillation_losses = []

        fusion_accuracy = utils.AverageMeter()
        fusion_ce_loss = utils.AverageMeter()
        fusion_ensemble_loss = utils.AverageMeter()
        fusion_loss = utils.AverageMeter()

        leader_accuracy = utils.AverageMeter()
        leader_ce_loss = utils.AverageMeter()
        leader_ensemble_loss = utils.AverageMeter()
        leader_self_distillation_feature_loss = utils.AverageMeter()
        leader_self_distillation_attention_loss = utils.AverageMeter()
        leader_self_distillation_loss = utils.AverageMeter()
        leader_fusion_loss = utils.AverageMeter()
        leader_trans_fusion_loss = utils.AverageMeter()
        leader_loss = utils.AverageMeter()

        average_accuracy = utils.AverageMeter()
        ensemble_accuracy = utils.AverageMeter()

        self.fusion_module.train()
        self.leader_model.train()
        for i in range(self.model_num):
            self.models[i].train()
            losses.append(utils.AverageMeter())
            ce_losses.append(utils.AverageMeter())
            dml_losses.append(utils.AverageMeter())
            diversity_losses.append(utils.AverageMeter())
            self_distillation_feature_losses.append(utils.AverageMeter())
            self_distillation_attention_losses.append(utils.AverageMeter())
            self_distillation_losses.append(utils.AverageMeter())
            accuracy.append(utils.AverageMeter())

        print_freq = len(self.trainLoader.dataset) // self.opt.train_batch_size // 10
        start_time = time.time()
        dataset_size = len(self.trainLoader.dataset)
        epoch_iter = 0

        for batch, (inputs, labels) in enumerate(self.trainLoader):

            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.adjust_learning_rates(epoch, batch, dataset_size // self.train_batch_size)
            
            epoch_iter += self.train_batch_size

            ensemble_output = 0.0
            outputs = []
            total_feature_maps = []
            fusion_module_inputs = []
            leader_output, leader_trans_fusion_output = self.leader_model(inputs)
            for i in range(self.model_num):
                outputs.append(self.models[i](inputs))
                ensemble_output += outputs[-1]
                total_feature_maps.append(list(self.models[i].total_feature_maps.values()))
                fusion_module_inputs.append(list(self.models[i].total_feature_maps.values())[-1].detach())
            fusion_module_inputs = torch.cat(fusion_module_inputs, dim=1)
            fusion_output = self.fusion_module(fusion_module_inputs)
            ensemble_output = ensemble_output / self.model_num

            # backward models
            for i in range(self.model_num):

                loss_ce = self.criterion_CE(outputs[i], labels)
                loss_dml = 0.0

                for j in range(self.model_num):
                    if i != j:
                        loss_dml += self.criterion_KL(F.log_softmax(outputs[i] / self.temperature, dim=1),
                                                      F.softmax(outputs[j].detach() / self.temperature, dim=1))

                if i != 0:
                    current_attention_map = total_feature_maps[i][-1].pow(2).mean(1, keepdim=True)
                    other_attention_map = total_feature_maps[i - 1][-1].detach().pow(2).mean(1, keepdim=True)
                    loss_diversity = self.lambda_diversity * self.diversity_loss(current_attention_map,
                                                                                 other_attention_map)
                    loss_self_distllation = self.lambda_diversity * \
                                            self.self_distillation_loss(self.sd_models[i - 1],
                                                                        total_feature_maps[i],
                                                                        input_feature_map=self.diversity_target(
                                                                            total_feature_maps[i - 1][-1].detach()))
                else:
                    loss_diversity = 0.0
                    loss_self_distllation = 0.0
                loss_dml = (self.temperature ** 2) * loss_dml / (self.model_num - 1)
                loss = loss_ce + loss_dml + loss_diversity + loss_self_distllation

                # measure accuracy and record loss
                prec = utils.accuracy(outputs[i].data, labels.data, topk=topk)
                losses[i].update(loss.item(), inputs.size(0))
                ce_losses[i].update(loss_ce.item(), inputs.size(0))
                dml_losses[i].update(loss_dml, inputs.size(0))
                diversity_losses[i].update(loss_diversity, inputs.size(0))
                self_distillation_losses[i].update(loss_self_distllation, inputs.size(0))
                accuracy[i].update(prec[0], inputs.size(0))

                self.optimizers[i].zero_grad()
                loss.backward()
                self.optimizers[i].step()

            # backward fusion module
            loss_fusion_ce = self.criterion_CE(fusion_output, labels)
            loss_fusion_ensemble = (self.temperature ** 2) * self.criterion_KL(
                F.log_softmax(fusion_output / self.temperature, dim=1),
                F.softmax(ensemble_output.detach() / self.temperature, dim=1))
            loss_fusion = loss_fusion_ce + loss_fusion_ensemble
            self.fusion_optimizer.zero_grad()
            loss_fusion.backward()
            self.fusion_optimizer.step()

            fusion_ce_loss.update(loss_fusion_ce.item(), inputs.size(0))
            fusion_ensemble_loss.update(loss_fusion_ensemble.item(), inputs.size(0))
            fusion_loss.update(loss_fusion.item(), inputs.size(0))
            fusion_prec = utils.accuracy(fusion_output, labels.data, topk=topk)
            fusion_accuracy.update(fusion_prec[0], inputs.size(0))

            # backward leader model
            leader_feature_maps = list(self.leader_model.total_feature_maps.values())
            loss_leader_ce = self.criterion_CE(leader_output, labels)
            loss_leader_ensemble = (self.temperature ** 2) * self.criterion_KL(
                F.log_softmax(leader_output / self.temperature, dim=1),
                F.softmax(fusion_output.detach() / self.temperature, dim=1))
            loss_leader_fusion = self.lambda_fusion * self.fusion_loss(
                leader_feature_maps[-1].pow(2).mean(1, keepdim=True),
                list(self.fusion_module.total_feature_maps.values())[-1].detach().pow(2).mean(1, keepdim=True))
            loss_leader_trans_fusion = self.lambda_fusion * \
                                       self.fusion_loss(leader_trans_fusion_output.pow(2).mean(1, keepdim=True),
                                                           fusion_module_inputs.pow(2).mean(1, keepdim=True))
            loss_leader_self_distillation = self.lambda_fusion * \
                                            self.self_distillation_loss(self.sd_leader_model, leader_feature_maps,
                                                                        input_feature_map=list(
                                                                            self.fusion_module.total_feature_maps.values())[
                                                                            -1].detach())
            loss_leader = loss_leader_ce + loss_leader_ensemble + loss_leader_fusion + loss_leader_trans_fusion + loss_leader_self_distillation

            self.leader_optimizer.zero_grad()
            loss_leader.backward()
            self.leader_optimizer.step()

            leader_ce_loss.update(loss_leader_ce.item(), inputs.size(0))
            leader_ensemble_loss.update(loss_leader_ensemble.item(), inputs.size(0))
            leader_fusion_loss.update(loss_leader_fusion, inputs.size(0))
            leader_trans_fusion_loss.update(loss_leader_trans_fusion, inputs.size(0))
            leader_self_distillation_loss.update(loss_leader_self_distillation, inputs.size(0))
            leader_loss.update(loss_leader.item(), inputs.size(0))
            leader_prec = utils.accuracy(leader_output, labels.data, topk=topk)
            leader_accuracy.update(leader_prec[0], inputs.size(0))

            # update self distillation model after all models updated
            for i in range(1, self.model_num):
                loss_self_distillation_feature, loss_self_distillation_attention = \
                    self.train_self_distillation_model(self.sd_models[i - 1],
                                                       self.sd_optimizers[i - 1],
                                                       target_feature_maps=total_feature_maps[i])
                self_distillation_feature_losses[i].update(loss_self_distillation_feature, inputs.size(0))
                self_distillation_attention_losses[i].update(loss_self_distillation_attention, inputs.size(0))

            loss_leader_self_distillation_feature, loss_leader_self_distillation_attention = \
                self.train_self_distillation_model(self.sd_leader_model,
                                                   self.sd_leader_optimizer,
                                                   target_feature_maps=leader_feature_maps)
            leader_self_distillation_feature_loss.update(loss_leader_self_distillation_feature, inputs.size(0))
            leader_self_distillation_attention_loss.update(loss_leader_self_distillation_attention, inputs.size(0))

            average_prec = utils.average_accuracy(outputs, labels.data, topk=topk)
            ensemble_prec = utils.ensemble_accuracy(outputs, labels.data, topk=topk)

            average_accuracy.update(average_prec[0], inputs.size(0))
            ensemble_accuracy.update(ensemble_prec[0], inputs.size(0))

            if batch % print_freq == 0 and batch != 0:
                current_time = time.time()
                cost_time = current_time - start_time

                msg = 'Epoch[{}] ({}/{})\tTime {:.2f}s\t'.format(
                    epoch, batch * self.train_batch_size, len(self.trainLoader.dataset), cost_time)
                for i in range(self.model_num):

                    msg += '|Model[{}]: Loss:{:.4f}\t' \
                           'CE Loss:{:.4f}\tDML Loss:{:.4f}\t' \
                           'Diversity Loss:{:.4f}\tSD Feature:{:.4f}' \
                           'SD Attention:{:.4f}\tSelf Distillation Loss:{:.4f}\t' \
                           'Accuracy {:.2f}%\t'.format(
                        i, float(losses[i].avg), float(ce_losses[i].avg), float(dml_losses[i].avg),
                        float(diversity_losses[i].avg), float(self_distillation_feature_losses[i].avg),
                        float(self_distillation_attention_losses[i].avg), float(self_distillation_losses[i].avg),
                        float(accuracy[i].avg))
                msg += '|Model[{}]: Loss:{:.4f}\t' \
                       'CE Loss:{:.4f}\tKL Loss:{:.4f}\t' \
                       'Accuracy {:.2f}%\t'.format(
                    'fusion', float(fusion_loss.avg), float(fusion_ce_loss.avg), float(fusion_ensemble_loss.avg),
                    float(fusion_accuracy.avg))
                msg += '|Model[{}]: Loss:{:.4f}\t' \
                       'CE Loss:{:.4f}\tEnsemble Loss:{:.4f}\t' \
                       'Fusion Loss:{:.4f}\tTrans Fusion Loss:{:.4f}\t' \
                       'SD Feature:{:.4f}\tSD Attention:{:.4f}\t' \
                       'Self Distillation Loss:{:.4f}\tAccuracy {:.2f}%\t'.format(
                    'leader', float(leader_loss.avg), float(leader_ce_loss.avg),
                    float(leader_ensemble_loss.avg), float(leader_fusion_loss.avg), float(leader_trans_fusion_loss.avg),
                    float(leader_self_distillation_feature_loss.avg),
                    float(leader_self_distillation_attention_loss.avg),
                    float(leader_self_distillation_loss.avg), float(leader_accuracy.avg))

                msg += '|Average Acc:{:.2f}\tEnsemble Acc:{:.2f}'.format(float(average_accuracy.avg),
                                                                         float(ensemble_accuracy.avg))
                self.logger.info(msg)

                start_time = current_time

    def test(self, epoch, topk=(1,)):

        losses = []
        accuracy = []
        top5_accuracy = []
        fusion_accuracy = utils.AverageMeter()
        leader_accuracy = utils.AverageMeter()
        average_accuracy = utils.AverageMeter()
        ensemble_accuracy = utils.AverageMeter()
        self.fusion_module.eval()
        self.leader_model.eval()
        for i in range(self.model_num):
            self.models[i].eval()
            accuracy.append(utils.AverageMeter())
            top5_accuracy.append(utils.AverageMeter())
        accuracy.append(fusion_accuracy)
        accuracy.append(leader_accuracy)

        start_time = time.time()
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(self.testLoader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = []
                fusion_module_inputs = []
                leader_output, _ = self.leader_model(inputs)
                for i in range(self.model_num):
                    outputs.append(self.models[i](inputs))
                    fusion_module_inputs.append(list(self.models[i].total_feature_maps.values())[-1].detach())
                fusion_module_inputs = torch.cat(fusion_module_inputs, dim=1)
                fusion_output = self.fusion_module(fusion_module_inputs)

                # measure accuracy and record loss
                for i in range(self.model_num):
                    prec = utils.accuracy(outputs[i].data, labels.data, topk=topk)
                    accuracy[i].update(prec[0], inputs.size(0))
                    if len(topk) == 2:
                        top5_accuracy[i].update(prec[1], inputs.size(0))

                fusion_prec = utils.accuracy(fusion_output, labels.data, topk=topk)
                fusion_accuracy.update(fusion_prec[0], inputs.size(0))

                leader_prec = utils.accuracy(leader_output, labels.data, topk=topk)
                leader_accuracy.update(leader_prec[0], inputs.size(0))

                average_prec = utils.average_accuracy(outputs, labels.data, topk=topk)
                ensemble_prec = utils.ensemble_accuracy(outputs, labels.data, topk=topk)

                average_accuracy.update(average_prec[0], inputs.size(0))
                ensemble_accuracy.update(ensemble_prec[0], inputs.size(0))

            current_time = time.time()

            msg = 'Epoch[{}]\tTime {:.2f}s\t'.format(epoch, current_time - start_time)

            for i in range(self.model_num):
                msg += 'Model[{}]:\tAccuracy {:.2f}%\t'.format(i, float(accuracy[i].avg))
            msg += 'Model[{}]:\tAccuracy {:.2f}%\t'.format('Fusion', float(fusion_accuracy.avg))
            msg += 'Model[{}]:\tAccuracy {:.2f}%\t'.format('Leader', float(leader_accuracy.avg))

            msg += 'Average Acc:{:.2f}\tEnsemble Acc:{:.2f}'.format(float(average_accuracy.avg),
                                                                    float(ensemble_accuracy.avg))

            self.logger.info(msg + '\n')

        return losses, accuracy, top5_accuracy, average_accuracy, ensemble_accuracy

    def train_self_distillation_model(self, sd_model, sd_optimizer, target_feature_maps):

        sd_model.train()
        sd_feature_loss = 0.0
        sd_attention_loss = 0.0
        input = target_feature_maps[-1].detach()
        sd_model(input)
        total_feature_maps = list(sd_model.total_feature_maps.values())
        total_feature_maps.reverse()

        for i, feature_map in enumerate(total_feature_maps):
            attention_map = feature_map.pow(2).mean(1, keepdim=True)
            target_attenion_map = target_feature_maps[i].detach().pow(2).mean(1, keepdim=True)

            sd_feature_loss += self.lambda_self_distillation * \
                               self.attention_loss(feature_map,
                                                   target_feature_maps[i].detach())
            sd_attention_loss += self.lambda_self_distillation * \
                                 self.attention_loss(attention_map,
                                                     target_attenion_map)

        sd_loss = sd_feature_loss + sd_attention_loss

        sd_optimizer.zero_grad()
        sd_loss.backward()
        sd_optimizer.step()

        return sd_feature_loss, sd_attention_loss

    def self_distillation_loss(self, sd_model, source_feature_maps, input_feature_map=None):

        sd_model.eval()
        sd_loss = 0.0

        if input_feature_map is None:
            input_feature_map = source_feature_maps[-1].detach()
        else:
            input_feature_map = input_feature_map.detach()
        sd_model(input_feature_map)
        target_feature_maps = list(sd_model.total_feature_maps.values())
        target_feature_maps.reverse()

        for i, feature_map in enumerate(target_feature_maps):
            source_attention_map = source_feature_maps[i].pow(2).mean(1, keepdim=True)
            target_attention_map = feature_map.detach().pow(2).mean(1, keepdim=True)
            sd_loss += self.attention_loss(source_attention_map, target_attention_map)

        return sd_loss

    def lambda_warmup(self, epoch):

        def warmup(lambda_coeff, epoch, alpha=5):

            if epoch <= alpha:
                return lambda_coeff * math.exp(-5 * math.pow((1 - float(epoch) / alpha), 2))
            else:
                return lambda_coeff

        self.lambda_diversity = warmup(self.opt.lambda_diversity, epoch)
        self.lambda_fusion = warmup(self.opt.lambda_fusion, epoch)
        self.lambda_self_distillation = warmup(self.opt.lambda_self_distillation, epoch)

    def diversity_target(self, y):

        attention_y = y.pow(2).mean(1, keepdim=True)
        attention_y_size = attention_y.size()
        norm_y = torch.norm(attention_y.view(attention_y.size(0), -1), dim=1, keepdim=True)
        attention_y = F.normalize(attention_y.view(attention_y.size(0), -1))
        threshold = attention_y.topk(int(attention_y.size(1) / 3), largest=False)[0][:, -1].unsqueeze(-1)
        target_y = (norm_y / 2 - attention_y) * torch.sign(attention_y - threshold) + norm_y / 2
        diff = (target_y - attention_y.view(attention_y.size(0), -1))
        return y + ((diff * norm_y / y.size(1)).view(attention_y_size))

    def diversity_loss(self, x, y):

        norm_y = torch.norm(y.view(y.size(0), -1), dim=1, keepdim=True)
        x = F.normalize(x.view(x.size(0), -1))
        y = F.normalize(y.view(y.size(0), -1))
        threshold = y.topk(int(y.size(1) / 3), largest=False)[0][:, -1].unsqueeze(-1)
        y = (norm_y / 2 - y) * torch.sign(y - threshold) + norm_y / 2
        return (x - y).pow(2).mean()

    def fusion_loss(self, x, y):

        x = F.normalize(x.view(x.size(0), -1))
        y = F.normalize(y.view(y.size(0), -1))
        return (x - y).pow(2).mean()

    def attention_loss(self, x, y):

        x = F.normalize(x.view(x.size(0), -1))
        y = F.normalize(y.view(y.size(0), -1))
        return (x - y).pow(2).mean()

    def load_models(self, model, opt):

        if opt.load_path is None or not os.path.exists(opt.load_path):
            raise FileExistsError('Load path must be exist!!!')
        ckpt = torch.load(opt.load_path, map_location=self.device)
        model.load_state_dict(ckpt['weight'])

    def save_models(self, model, epoch, name, opt, isbest):

        save_dir = os.path.join(opt.checkpoints_dir, opt.name, 'checkpoints')
        utils.mkdirs(save_dir)
        ckpt = {
            'weight': model.state_dict(),
            'epoch': epoch,
            'cfg': opt.model,
            'index': name
        }
        if isbest:
            torch.save(ckpt, os.path.join(save_dir, 'model%s_best.pth' % name))
        else:
            torch.save(ckpt, os.path.join(save_dir, 'model%s_%d.pth' % (name, epoch)))

    def adjust_learning_rates(self, epoch, step, len_epoch):

        def adjust_lr(optimizer, epoch, step, len_epoch):

            factor = epoch // 30

            if epoch >= 80:
                factor = factor + 1

            lr = self.opt.lr * (0.1 ** factor)

            # Warmup
            if epoch < 5:
                lr = lr * float(1 + step + epoch * len_epoch) / (5. * len_epoch)

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        for i in range(self.model_num):
            adjust_lr(self.optimizers[i], epoch, step, len_epoch)
        adjust_lr(self.leader_optimizer, epoch, step, len_epoch)
        adjust_lr(self.fusion_optimizer, epoch, step, len_epoch)