from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_gmm_distribution(gmm, input_loss, output_fig_name):
    # plot probability distribution
    x_range = np.linspace(input_loss.min().item()-0.1, input_loss.max().item()+0.1, 200).reshape(-1, 1)
    GMM_curves = gmm.predict_proba(x_range)
    fig, ax1 = plt.subplots()
    for i in range(GMM_curves.shape[1]):
        labeled_curve = GMM_curves[:,i]
        ax1.plot(x_range, labeled_curve, label=f"component {i}")
        ax1.set_xlabel("Loss")
        ax1.set_ylabel("Probability")
        ax1.legend()
        ax1.tick_params(axis='y')

    # plot loss histogram
    ax2 = ax1.twinx()
    ax2.hist(input_loss.numpy().flatten(), alpha=0.5, bins=25)
    ax2.set_ylabel("Counts")
    ax2.tick_params(axis='y')
    
    fig.suptitle(f"GMM mixture, component considered as clean: {gmm.means_.argmin()}")
    fig.tight_layout()
    plt.savefig(output_fig_name)
    plt.close()

def plot_tsnes(eval_dataset, all_targets, all_features, prob, epoch, model_name):
    # run tsne on feature points
    all_targets = torch.cat(all_targets, dim=0).cpu().numpy()
    all_features = torch.cat(all_features, dim=0).cpu().numpy()
    # tsne_features = all_features[:, :2]     # fake tsne to debug faster
    tsne_features = TSNE(n_components=2, perplexity=30).fit_transform(all_features)
    tsne_features = tsne_features.astype(np.float64)    # for json serialization

    idx_to_class = eval_dataset.get_idx_to_class_map()
    data_paths = eval_dataset.get_data_paths()

    all_clean_indices = (prob > args.p_threshold).nonzero()[0]
    nrows = 3
    ncols = 4
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(32, 24))
    combined_fig, combined_ax = plt.subplots(figsize=(16, 12))
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', '#014d4e'] # 11 colors for 11 classes
    tsne_results = []
    for class_number, class_name in idx_to_class.items():
        class_indices = (all_targets == class_number).nonzero()[0]
        class_clean_indices = np.intersect1d(all_clean_indices, class_indices)
        class_noisy_indices = np.setdiff1d(class_indices, class_clean_indices)
        class_clean_features = tsne_features[class_clean_indices, :]
        class_noisy_features = tsne_features[class_noisy_indices, :]

        # plot clean/noisy on a separate plot, and also on the combined plot
        for axis in (ax[class_number // ncols, class_number % ncols], combined_ax):
            axis.scatter(class_clean_features[:, 0], class_clean_features[:, 1], marker=".", color=colors[class_number])
            axis.scatter(class_noisy_features[:, 0], class_noisy_features[:, 1], marker="x", color=colors[class_number])
            axis.plot([], [], label=class_name, color=colors[class_number])  # add label separately to cover both clean and noisy
        ax[class_number // ncols, class_number % ncols].set_title(class_name)

        # add this info to the json for interactive visualization
        for i in class_clean_indices:
            tsne_results.append({
                "class": class_name,
                "x": tsne_features[i, 0],
                "y": tsne_features[i, 1],
                "noisy": False,
                "filepath": data_paths[i]
            })
        for i in class_noisy_indices:
            tsne_results.append({
                "class": class_name,
                "x": tsne_features[i, 0],
                "y": tsne_features[i, 1],
                "noisy": True,
                "filepath": data_paths[i]
            })

    combined_ax.set_title(f"TSNE on ResNet18 before output layer, n={all_targets.shape[0]}, X = noisy, O = clean")
    combined_ax.legend()
    fig.savefig(f"{args.output_dir}/epoch_{epoch}/{model_name}_tsne.png")
    combined_fig.savefig(f"{args.output_dir}/epoch_{epoch}/{model_name}_tsne_combined.png")
    plt.close(fig)
    plt.close(combined_fig)

    with open(f"{args.output_dir}/epoch_{epoch}/{model_name}_tsne_data.json", "w") as f:
        tsne_dict = {}
        tsne_dict['data'] = tsne_results
        json.dump(tsne_dict, f)


def plot_confusion_matrices():
    # relating to the confusion matrix
    total_predictions = torch.hstack(total_predictions).detach().cpu().numpy()
    total_targets = torch.hstack(total_targets).detach().cpu().numpy()
    classes = [idx_to_class[i] for i in range(args.num_class)]

    # create and plot a confusion matrix
    cm = confusion_matrix(total_targets, total_predictions, labels=range(args.num_class))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

    cm_norm = confusion_matrix(total_targets, total_predictions, labels=range(args.num_class), normalize='true')
    disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=classes)

    os.makedirs(f"{args.output_dir}/epoch_{epoch}", exist_ok=True)
    fig, ax = plt.subplots(figsize=(16, 14))
    ax.tick_params(axis='both', which='major', labelsize=13)
    disp.plot(ax=ax, xticks_rotation='vertical')
    plt.savefig(f"{args.output_dir}/epoch_{epoch}/confusion_table.png")
    plt.close()
    
    fig, ax = plt.subplots(figsize=(16, 14))
    ax.tick_params(axis='both', which='major', labelsize=13)
    disp_norm.plot(ax=ax, xticks_rotation='vertical')
    plt.savefig(f"{args.output_dir}/epoch_{epoch}/confusion_table_norm.png")
    plt.close()