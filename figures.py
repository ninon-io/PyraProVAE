#%% -*- coding: utf-8 -*-

import matplotlib
import torch
import torch.nn as nn
import os.path
import argparse
import numpy as np
# Plotting
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
from symbolic import symbolic_features, features_simple
"""
###################


AE-specific evaluation 
    - Latent space 
    - Semantic parameter 
    - Meta-parameters 
    - AE reconstruction


###################
"""

"""	
###################	
Dimensions evaluation for AE models	
###################	
"""	
def evaluate_dimensions(model, test_loader, pca=None, latent_dims = 16, n_steps = 40, pos=[-1, 0, 1], name=''):	
    print('[Evaluate latent dimensions.]')	
    if (pca is not None):
        latent_dims = pca.n_features_
    # Create a latent vector
    var_z = torch.linspace(-4, 4, n_steps)
    for l in range(latent_dims):
        # Create figure per dimension
        fig = plt.figure(figsize=(20, 10))
        outer = gridspec.GridSpec(2, 1, wspace=0.2, hspace=0.4)
        print('   - Dimension ' + str(l))	
        fake_batch = torch.zeros(n_steps, latent_dims)	
        fake_batch[:, l] = var_z
        if (pca is not None):
            fake_batch = torch.Tensor(pca.inverse_transform(fake_batch))	
        # Generate VAE outputs	
        out = model.decode(fake_batch)
        if (len(out.shape) > 3):
            out = torch.argmax(out, dim=1)
        print(out.shape)
        # Compute symbolic descriptors	
        desc = torch.zeros(n_steps, len(features_simple))
        for i, x_cur in enumerate(out):
            cur_feat = symbolic_features(x_cur, feature_set=features_simple)
            for f_i, f in enumerate(features_simple.keys()):
                desc[i, f_i] = cur_feat[f]
        #desc /= torch.max(desc, dim=0)
        ax = plt.Subplot(fig, outer[0])
        ax.plot(desc)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.legend(features_simple.keys())
        fig.add_subplot(ax)
        inner = gridspec.GridSpecFromSubplotSpec(1, 10, subplot_spec=outer[1], wspace=0.1, hspace=0.1)
        for i in range(10):
            ax = plt.Subplot(fig, inner[i])
            ax.matshow(out[i * 4], aspect='auto')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            fig.add_subplot(ax)
        plt.savefig(name + str(l) + '.pdf')
        plt.close()      
                
"""
###################
Meta-parameters evaluation for AE models
###################
"""
def evaluate_tranlations(model, test_loader, pca=None, latent_dims = 16, n_steps = 40, pos=[-1, 0, 1], name=''):	
    print('[Evaluate latent dimensions.]')	
    if (pca is not None):
        latent_dims = pca.n_features_
    # Select 8 random points from the test set
    fixed_data = next(iter(test_loader))
    n_examples = 8
    in_data = fixed_data[np.random.randint(0, fixed_data.shape[0], size=(8))].to(args.device)
    # Create a latent vector
    var_z = torch.linspace(-4, 4, n_steps)
    for ex in range(n_examples):
        for l in range(latent_dims):
            # Create figure per dimension
            fig = plt.figure(figsize=(20, 10))
            outer = gridspec.GridSpec(2, 1, wspace=0.2, hspace=0.4)
            print('   - Example ' + str(l))	
            fake_batch = torch.zeros(n_steps, latent_dims)	
            fake_batch[:, l] = var_z
            if (pca is not None):
                fake_batch = torch.Tensor(pca.inverse_transform(fake_batch))	
            # Generate VAE outputs	
            out = model.decode(fake_batch)
            if (len(out.shape) > 3):
                out = torch.argmax(out, dim=1)
            print(out.shape)
            # Compute symbolic descriptors	
            desc = torch.zeros(n_steps, len(features_simple))
            for i, x_cur in enumerate(out):
                cur_feat = symbolic_features(x_cur, feature_set=features_simple)
                for f_i, f in enumerate(features_simple.keys()):
                    desc[i, f_i] = cur_feat[f]
            #desc /= torch.max(desc, dim=0)
            ax = plt.Subplot(fig, outer[0])
            ax.plot(desc)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.legend(features_simple.keys())
            fig.add_subplot(ax)
            inner = gridspec.GridSpecFromSubplotSpec(1, 10, subplot_spec=outer[1], wspace=0.1, hspace=0.1)
            for i in range(10):
                ax = plt.Subplot(fig, inner[i])
                ax.matshow(out[i * 4], aspect='auto')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                fig.add_subplot(ax)
            plt.savefig(name + str(l) + '_' + str(ex) '.pdf')
            plt.close()    
        
def evaluate_translations(model, test_loader, args, train=False, name=None, n_recons = 8, n_steps = 100):
    print('  - Evaluate meta parameters.')
    latent_dims = model.ae_model.latent_dims
    fig = plt.figure(figsize=(10, 20))
    outer = gridspec.GridSpec(latent_dims + 1, 3, wspace=0.2, hspace=0.4)
    # Select 5 random points from the test set
    fixed_data, fixed_params, fixed_meta, fixed_audio = next(iter(test_loader))
    in_data = fixed_data[np.random.randint(0, fixed_data.shape[0], size=(32))].to(args.device)
    # Find corresponding params
    _, in_data, _ = model.ae_model(in_data)
    if (args.semantic_dim > -1):
        in_data, _ = model.disentangling(in_data)
    z_var = 0
    for l in range(latent_dims):
        var_z = torch.linspace(-4, 4, n_steps)
        fake_batch = torch.zeros(n_steps, latent_dims)
        fake_batch[:, l] = var_z
        fake_batch = fake_batch.to(args.device)
        # Generate VAE outputs
        x_tilde_full = model.ae_model.decode(fake_batch)
        # Perform regression
        out = model.regression_model(fake_batch)
        if (args.loss in ['multinomial']):
            tmp = out.view(out.shape[0], -1, latent_dims).max(dim=1)[1]
            out = tmp.float() / (args.n_classes - 1.)
        if (args.loss in ['multi_mse']):
            out = out.view(out.shape[0], -1, latent_dims)
            out = out[:, -1, :]
        # Select parameters
        var_param = out.std(dim=0)
        idx = torch.argsort(var_param, descending=True)
        # To keep coloring consistent we blank out all parameters above 5 most varying
        out[:, idx[5:]] = torch.zeros(out.shape[0], len(idx[5:])).to(out.device)
        ax = plt.Subplot(fig, outer[l*3])
        ax.plot(out.detach().cpu().numpy())
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if (hasattr(args, 'z_vars')):
            z_var = args.z_vars[l].item()
        ax.set_title('$z_{' + str(l) + '}$ - %.2f - %.3f'%((z_var), (var_param[idx[:5]].mean().item())))
        fig.add_subplot(ax)
        # Reconstruct a handful of points
        fake_batch = torch.zeros(n_recons, latent_dims)
        fake_batch[:, l] = torch.linspace(-4, 4, n_recons)
        fake_batch = fake_batch.to(args.device)
        # Reconstruct with the VAE
        x_tilde = model.ae_model.decode(fake_batch)
        # Reconstruct with the synth engine
        if (args.synthesize == True and train == False and ((var_param[idx[:5]].mean().item() > 0.15) or ((args.semantic_dim > -1) and (l == 0)))):
            out_batch = model.regression_model(fake_batch)
            if (args.loss in ['multinomial']):
                tmp = out_batch.view(out_batch.shape[0], -1, args.latent_dims).max(dim=1)[1]
                out_batch = tmp.float() / (args.n_classes - 1.)
            if (args.loss in ['multi_mse']):
                out_batch = out_batch.view(out_batch.shape[0], -1, args.latent_dims)
                out_batch = out_batch[:, -1, :]
            print('      - Generate audio for latent ' + str(l))
            from synth.synthesize import synthesize_batch
            audio = synthesize_batch(out_batch.cpu(), test_loader.dataset.final_params, args.engine, args.generator, args.param_defaults, args.rev_idx, orig_wave=None, name=None)
            save_batch_audio(audio, args.base_audio + '_meta_parameters_z' + str(l) + '_v' + str(var_param[idx[:5]].mean().item()))
            # Now check how this parameter act on various sounds
            n_ins = ((args.semantic_dim > -1) and (l == 0)) and 32 or 4
            for s in range(n_ins):
                print('          - Generate audio for meta-modified ' + str(s))
                tmp_data = in_data[s].clone().unsqueeze(0).repeat(n_recons, 1)
                tmp_data[:, l] = torch.linspace(-4, 4, n_recons)
                tmp_data = model.regression_model(tmp_data)
                if (args.loss in ['multinomial']):
                    tmp = tmp_data.view(tmp_data.shape[0], -1, args.latent_dims).max(dim=1)[1]
                    tmp_data = tmp.float() / (args.n_classes - 1.)
                if (args.loss in ['multi_mse']):
                    tmp_data = tmp_data.view(tmp_data.shape[0], -1, args.latent_dims)
                    tmp_data = tmp_data[:, -1, :]
                # Synthesize meta-modified test example :)                
                audio = synthesize_batch(tmp_data.cpu(), test_loader.dataset.final_params, args.engine, args.generator, args.param_defaults, args.rev_idx, orig_wave=None, name=None)
                save_batch_audio(audio, args.base_audio + '_meta_parameters_z' + str(l) + '_b' + str(s))
        if len(x_tilde.shape) > 3:
            x_tilde = x_tilde[:,0]
        inner = gridspec.GridSpecFromSubplotSpec(1, 8,
            subplot_spec=outer[l*3+1], wspace=0.1, hspace=0.1)
        for n in range(n_recons):
            ax = plt.Subplot(fig, inner[n])
            ax.imshow(x_tilde[n].detach().cpu().numpy(), aspect='auto')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            fig.add_subplot(ax)
        # Unscale and un-log output
        x_tilde_full = (x_tilde_full * test_loader.dataset.vars["mel"]) + test_loader.dataset.means["mel"]
        if (args.data in ['mel',"mel_mfcc"]):
            x_tilde_full = torch.exp(x_tilde_full)
        x_tilde_full = x_tilde_full[:,0]
        # Compute descriptors
        descs = compute_descriptors(x_tilde_full.detach().cpu().numpy())
        ax = plt.Subplot(fig, outer[l*3+2])
        ax.plot(descs)
        fig.add_subplot(ax)
    # Just fake plots for legends
    fake = torch.linspace(1, len(idx), len(idx)).repeat(out.shape[0], 1)
    ax = plt.Subplot(fig, outer[latent_dims*3])
    ax.plot(fake.numpy())
    ax.legend(test_loader.dataset.final_params)
    fig.add_subplot(ax)
    fake = torch.linspace(1, len(descriptors), len(descriptors)).repeat(out.shape[0], 1)
    ax = plt.Subplot(fig, outer[latent_dims*3+2])
    ax.plot(fake.numpy())
    ax.legend(descriptors)
    fig.add_subplot(ax)
    # Just generate a legend for kicks
    if (name is not None):
        plt.savefig(name + '_meta_parameters.pdf')
        plt.close()
    if (train == False and name is None):
        plt.savefig(args.base_img + '_meta_parameters.pdf')
        plt.close()
"""
###################
Evaluate latent neighborhoods 
###################
"""
def evaluate_latent_neighborhood(model, test_loader, args, train=False, name=None):
    from synth.synthesize import synthesize_batch
    print('  - Evaluate latent neighborhoods.')
    cur_batch = 0
    for (x, y, _, x_wave) in test_loader:
        if (cur_batch > 8):
            break
        # Send to device
        x, y = x.to(args.device), y.to(args.device)
        # Encode our fixed batch
        _, out, _ = model.ae_model(x)
        if (args.semantic_dim > -1):
            out, _ = model.disentangling(out)
        print('  - Generate audio outputs (batch ' + str(cur_batch) + ').')
        # Select two random examples
        ids = [np.random.randint(0, x.shape[0]), np.random.randint(0, x.shape[0])]
        # Generate different local neighborhoods
        for i in [0, 1]:
            v_r = 0.5
            out1 = out[ids[i]] + (torch.randn(8, out.shape[1]) * v_r).to(args.device)
            out1 = model.regression_model(out1)
            if (args.loss in ['multinomial']):
                tmp = out1.view(out1.shape[0], -1, y.shape[1]).max(dim=1)[1]
                out1 = tmp.float() / (args.n_classes - 1.)
            if (args.loss in ['multi_mse']):
                out1 = out1.view(out1.shape[0], -1, y.shape[1])
                out1 = out1[:, -1, :]
            audio = synthesize_batch(out1.cpu(), test_loader.dataset.final_params, args.engine, args.generator, args.param_defaults, args.rev_idx, orig_wave=x_wave, name=None)
            save_batch_audio(audio, args.base_audio + '_neighbors_' + str(cur_batch) + '_p' + str(i))
            # Compute mel spectrograms
            full_mels = []
            for b in range(8):
                _, mse, sc, lm, f_mel = spectral_losses(audio[b], x[b], test_loader, args, raw=True)
                if (args.data == 'mel'):
                    f_mel = torch.log(f_mel + 1e-3)
                full_mels.append(f_mel.unsqueeze(0))
            full_mels = torch.cat(full_mels, dim=0)
            # Output batches comparisons
            if len(x.shape)>3: # get rid of mfcc
                x = x[:,0]
            id_full = [ids[i], 1, 2, 3, 4, 5, 6, 7]
            compare_batch_detailed(x[id_full].cpu(), y[id_full].cpu(), full_mels[:8].cpu().numpy(), out1[:8].detach().cpu(), None, x_wave[id_full].cpu(), audio[:8], name=args.base_img + '_neighbors_' + str(cur_batch) + '_' + str(i))
        # Create linear interpolation
        print('Perform interpolation')
        outs = torch.zeros(8, len(test_loader.dataset.param_names))
        for e in range(8):
            outs_t = model.regression_model(((out[ids[0]] * ((7.0-e)/7.0)) + (out[ids[1]] * (e/7.0))).unsqueeze(0))
            if (args.loss in ['multinomial']):
                tmp = outs_t.view(outs_t.shape[0], -1, y.shape[1]).max(dim=1)[1]
                outs_t = tmp.float() / (args.n_classes - 1.)
            if (args.loss in ['multi_mse']):
                outs_t = outs_t.view(outs_t.shape[0], -1, y.shape[1])
                outs_t = outs_t[:, -1, :]
            outs[e] = outs_t[0]
        # Compute mel spectrograms
        full_mels = []
        audio = synthesize_batch(outs.cpu(), test_loader.dataset.final_params, args.engine, args.generator, args.param_defaults, args.rev_idx, orig_wave=x_wave, name=None)
        save_batch_audio(audio, args.base_audio + '_neighbors_' + str(cur_batch) + '_interpolate')
        for b in range(outs.shape[0]):
            _, mse, sc, lm, f_mel = spectral_losses(audio[b], x[b], test_loader, args, raw=True)
            if (args.data == 'mel'):
                f_mel = torch.log(f_mel + 1e-3)
            full_mels.append(f_mel.unsqueeze(0))
        full_mels = torch.cat(full_mels, dim=0)
        # Output batches comparisons
        if len(x.shape)>3: # get rid of mfcc
            x = x[:,0]
        id_full = [ids[0], ids[1], 2, 3, 4, 5, 6, 7]
        compare_batch_detailed(x[id_full].cpu(), y[id_full].cpu(), full_mels[:8].cpu().numpy(), outs[:8].detach().cpu(), None, x_wave[id_full].cpu(), audio[:8], name=args.base_img + '_neighbors_' + str(cur_batch) + 'interpolate')
        cur_batch += 1            

"""
###################


Combined evaluations
    - Batch evaluation (during train)
    - Full final evaluation (end of train)
    - Model checking (for compiled results on same test set)


###################
"""

def sample2DSpace(vae, pca, cond, nbSamples, nbPlanes, Zp, Zc, figName=None):
    # First find boundaries of the space
    spaceBounds = np.zeros((3, 2))
    for i in range(3):
        spaceBounds[i, 0] = np.min(Zp[:, i])
        spaceBounds[i, 1] = np.max(Zp[:, i])
    # Now construct sampling grids for each axis
    samplingGrids = [None] * 3
    for i in range(3):
        samplingGrids[i] = np.meshgrid(np.linspace(-.9, .9, nbSamples), np.linspace(-.9, .9, nbSamples))
    # Create the set of planes
    planeDims = np.zeros((3, nbPlanes))
    for i in range(3):
        curVals = np.linspace(spaceBounds[i, 0], spaceBounds[i, 1], nbPlanes)
        for p in range(nbPlanes):
            planeDims[i, p] = curVals[p]
    dimNames = ['X', 'Y', 'Z'];
    for dim in range(3):
        print('Dimension ' + str(dim))
        curSampling = samplingGrids[dim]
        resultMatrix = {}
        for d in descriptors:
            resultMatrix[d] = [None] * nbPlanes
            for i in range(nbPlanes):
                resultMatrix[d][i] = np.zeros((nbSamples, nbSamples))
        for plane in range(nbPlanes):
            print('Plane ' + str(plane))
            curPlaneVal = planeDims[dim, plane]
            for x in range(nbSamples):
                for y in range(nbSamples):
                    if (dim == 0):
                        curPoint = [curPlaneVal, curSampling[0][x, y], curSampling[1][x, y]]
                    if (dim == 1):
                        curPoint = [curSampling[0][x, y], curPlaneVal, curSampling[1][x, y]]
                    if (dim == 2):
                        curPoint = [curSampling[0][x, y], curSampling[1][x, y], curPlaneVal]
                    descVals = sampleCompute(vae, torch.Tensor(curPoint), pca, cond, targetDims=[0, 1, 2])
                    for d in descriptors:
                        resultMatrix[d][plane][x, y] = descVals[d]
        plt.figure();
        for dI in range(len(descriptors)):
            d = descriptors[dI]
            for i in range(nbPlanes):
                plt.subplot(len(descriptors), nbPlanes, (dI * nbPlanes) + i + 1)
                plt.imshow(resultMatrix[d][i], interpolation="sinc");
                plt.tick_params(which='both', labelbottom=False, labelleft=False)
                if (i == 0):
                    plt.ylabel(d)   
        #plt.subplots_adjust(bottom=0.2, left=0.01, right=0.05, top=0.25)
        if (figName is not None):
            plt.savefig(figName+'_'+dimNames[dim]+'.png', bbox_inches='tight');
            plt.close()
            

def getDescriptorGrid(sampleGrid3D, vae, pca, cond):
    # Resulting sampling tensors
    point_hash = {}
    zs = np.zeros((np.ravel(sampleGrid3D[0]).shape[0], 3))
    current_idx = 0
    for x in range(sampleGrid3D[0].shape[0]):
        for y in range(sampleGrid3D[0].shape[1]):
            for z in range(sampleGrid3D[0].shape[2]):
                curPoint = [sampleGrid3D[0][x,y,z],sampleGrid3D[1][x,y,z],sampleGrid3D[2][x,y,z]]
                zs[current_idx] = np.array(curPoint)
                point_hash[(x,y,z)] = current_idx
                current_idx += 1
                
#    cond = vae.format_label_data(np.ones(zs.shape[0]))
    descVals = sampleBatchCompute(vae, zs, pca, cond)
    
    resultTensor = {}
    for d in descVals.keys():
        resultTensor[d] = np.zeros_like(sampleGrid3D[0])
    
    for x in range(sampleGrid3D[0].shape[0]):
        for y in range(sampleGrid3D[0].shape[1]):
            for z in range(sampleGrid3D[0].shape[2]):
                current_idx = point_hash[x,y,z]
                for d in descVals.keys():
                    resultTensor[d][x,y,z] = descVals[d][current_idx]

    return resultTensor
    
    
    
def sample3DSpace(vae, pca, cond, nbSamples, nbPlanes, Zp, Zc, figName=None, loadFrom=None, saveAs=None, resultTensor=None):
    # Create sampling grid
    samplingGrid3D = np.meshgrid(np.linspace(np.min(Zp[:, 0]), np.max(Zp[:, 0]), nbSamples),
                             np.linspace(np.min(Zp[:, 1]), np.max(Zp[:, 1]), nbSamples),
                             np.linspace(np.min(Zp[:, 2]), np.max(Zp[:, 2]), nbSamples))
    # Resulting sampling tensors
    if not loadFrom is None:
        print('loading from %s...'%loadFrom)
        resultTensor = np.load(loadFrom)[None][0]
    elif (resultTensor is None):
        resultTensor = getDescriptorGrid(samplingGrid3D, vae, pca, cond)
#        for d in descriptors:
#            resultTensor[d] = np.zeros((nbSamples, nbSamples, nbSamples))
#        for x in range(nbSamples):
#            for y in range(nbSamples):
#                for z in range(nbSamples):
#                    curPoint = [samplingGrid3D[0][x,y,z],samplingGrid3D[1][x,y,z],samplingGrid3D[2][x,y,z]]
#                    descVals = sampleCompute(vae, torch.Tensor(curPoint), pca, cond, targetDims=[0, 1, 2])
#                    for d in descriptors:
#                        resultTensor[d][x, y, z] = descVals[d]
    if not saveAs is None:
        print('saving as %s...'%saveAs)
        np.save(saveAs, resultTensor)
        
        
    axNames = ['X', 'Y', 'Z']
    # Sets of planes
    xVals = np.linspace(np.min(Zp[:, 0]), np.max(Zp[:, 0]), nbSamples)
    yVals = np.linspace(np.min(Zp[:, 1]), np.max(Zp[:, 1]), nbSamples)
    zVals = np.linspace(np.min(Zp[:, 2]), np.max(Zp[:, 2]), nbSamples)
    for dim in range(3):
        print('-- dim %d...'%dim)
        # For each descriptor
        for d in descriptors:
            print('descriptos %s...'%d)
            global i; i = 0;
            fig = plt.figure(figsize=(12, 6)) 
            gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1]) 
            ax = fig.add_subplot(gs[0], projection='3d')
            plt.title('Projection ' + axNames[dim] + ' - Spectral ' + d)
            if (dim == 0):
                surfYZ = np.array([[xVals[0], np.min(Zp[:, 1]), np.min(Zp[:, 2])],[xVals[0], np.max(Zp[:, 1]), np.min(Zp[:, 2])],
                                    [xVals[0], np.max(Zp[:, 1]), np.max(Zp[:, 2])],[xVals[0], np.min(Zp[:, 1]), np.max(Zp[:, 2])]])
            if (dim == 1):
                surfYZ = np.array([[np.min(Zp[:, 0]), yVals[0], np.min(Zp[:, 2])],[np.max(Zp[:, 0]), yVals[0], np.min(Zp[:, 2])],
                                    [np.max(Zp[:, 0]), yVals[0], np.max(Zp[:, 2])],[np.min(Zp[:, 0]), yVals[0], np.max(Zp[:, 2])]])
            if (dim == 2):
                surfYZ = np.array([[np.min(Zp[:, 0]), np.min(Zp[:, 1]), zVals[0]],[np.max(Zp[:, 0]), np.min(Zp[:, 1]), zVals[0]],
                                    [np.max(Zp[:, 0]), np.max(Zp[:, 1]), zVals[0]],[np.min(Zp[:, 0]), np.max(Zp[:, 1]), zVals[0]]])
            #ax.scatter(zLatent[:, 0], zLatent[:, 1], zLatent[:, 2])
            task = 'instrument'
            meta = np.array(audioSet.metadata[task])
            cmap = plt.cm.get_cmap('plasma', audioSet.classes[task]['_length'])
            c = []
            for j in meta:
                c.append(cmap(int(j)))   
            ax.scatter(Zp[:, 0], Zp[:,1], Zp[:, 2], c=Zc)
            lines = [None] * 4
            for j in range(4):
                lines[j], = ax.plot([surfYZ[j, 0], surfYZ[(j+1)%4, 0]], [surfYZ[j, 1], surfYZ[(j+1)%4, 1]], zs=[surfYZ[j, 2], surfYZ[(j+1)%4, 2]], linestyle='--', color='k', linewidth=2)
            for v in range(nbSamples):
                if (dim == 0):
                    surfYZ = np.array([[xVals[v], np.min(Zp[:, 1]), np.min(Zp[:, 2])],[xVals[v], np.max(Zp[:, 1]), np.min(Zp[:, 2])],
                                        [xVals[v], np.max(Zp[:, 1]), np.max(Zp[:, 2])],[xVals[v], np.min(Zp[:, 1]), np.max(Zp[:, 2])]])
                if (dim == 1):
                    surfYZ = np.array([[np.min(Zp[:, 0]), yVals[v], np.min(Zp[:, 2])],[np.max(Zp[:, 0]), yVals[v], np.min(Zp[:, 2])],
                                        [np.max(Zp[:, 0]), yVals[v], np.max(Zp[:, 2])],[np.min(Zp[:, 0]), yVals[v], np.max(Zp[:, 2])]])
                if (dim == 2):
                    surfYZ = np.array([[np.min(Zp[:, 0]), np.min(Zp[:, 1]), zVals[v]],[np.max(Zp[:, 0]), np.min(Zp[:, 1]), zVals[v]],
                                        [np.max(Zp[:, 0]), np.max(Zp[:, 1]), zVals[v]],[np.min(Zp[:, 0]), np.max(Zp[:, 1]), zVals[v]]])
                for j in range(4):
                    ax.plot([surfYZ[j, 0], surfYZ[(j+1)%4, 0]], [surfYZ[j, 1], surfYZ[(j+1)%4, 1]], zs=[surfYZ[j, 2], surfYZ[(j+1)%4, 2]], alpha=0.1, color='g', linewidth=2)
            ax1 = plt.subplot(gs[1])  
            if (dim == 0):
                im = ax1.imshow(resultTensor[d][i], animated=True)
            if (dim == 1):
                im = ax1.imshow(resultTensor[d][:, i, :], animated=True)
            if (dim == 2):
                im = ax1.imshow(resultTensor[d][:, :, i], animated=True)
            # Function to update
            def updatefig(*args):
                global i
                i += 1
                try:
                    if (dim == 0):
                        im.set_array(resultTensor[d][i])
                    if (dim == 1):
                        im.set_array(resultTensor[d][:, i, :])
                    if (dim == 2):
                        im.set_array(resultTensor[d][:, :, i])
                    if (dim == 0):
                        surfYZ = np.array([[xVals[i], np.min(Zp[:, 1]), np.min(Zp[:, 2])],[xVals[i], np.max(Zp[:, 1]), np.min(Zp[:, 2])],
                                            [xVals[i], np.max(Zp[:, 1]), np.max(Zp[:, 2])],[xVals[i], np.min(Zp[:, 1]), np.max(Zp[:, 2])]])
                    if (dim == 1):
                        surfYZ = np.array([[np.min(Zp[:, 0]), yVals[i], np.min(Zp[:, 2])],[np.max(Zp[:, 0]), yVals[i], np.min(Zp[:, 2])],
                                            [np.max(Zp[:, 0]), yVals[i], np.max(Zp[:, 2])],[np.min(Zp[:, 0]), yVals[i], np.max(Zp[:, 2])]])
                    if (dim == 2):
                        surfYZ = np.array([[np.min(Zp[:, 0]), np.min(Zp[:, 1]), zVals[i]],[np.max(Zp[:, 0]), np.min(Zp[:, 1]), zVals[i]],
                                            [np.max(Zp[:, 0]), np.max(Zp[:, 1]), zVals[i]],[np.min(Zp[:, 0]), np.max(Zp[:, 1]), zVals[i]]])
                    for j in range(4):
                        lines[j].set_data([surfYZ[j, 0], surfYZ[(j+1)%4, 0]], [surfYZ[j, 1], surfYZ[(j+1)%4, 1]])
                        lines[j].set_3d_properties([surfYZ[j, 2], surfYZ[(j+1)%4, 2]])
                except:
                    print('pass')
                return im,
            # Set up formatting for the movie files
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=15, metadata=dict(artist='acids.ircam.fr'), bitrate=1800)
            ani = animation.FuncAnimation(fig, updatefig, frames=nbSamples, interval=50, blit=True)
            ani.save(figName + '_' + d + '_' + axNames[dim] + '.mp4', writer=writer)
            ani.event_source.stop()
            del ani
            plt.close()
    return resultTensor
                
