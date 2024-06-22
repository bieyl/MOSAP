import numpy as np
import time
import os

from multiprocessing import Pool
from tqdm import tqdm
from keras.models import load_model, Model
from scipy.stats import gaussian_kde

from utils import *
from multiprocessing import Pool

def _aggr_output(x):
    return [np.mean(x[..., j]) for j in range(x.shape[-1])]


def _get_saved_path(base_path, dataset, dtype, layer_names):
    """Determine saved path of ats and pred

    Args:
        base_path (str): Base save path.
        dataset (str): Name of dataset.
        dtype (str): Name of dataset type (e.g., train, test, fgsm, ...).
        layer_names (list): List of layer names.

    Returns:
        ats_path: File path of ats.
        pred_path: File path of pred (independent of layers)
    """

    joined_layer_names = "_".join(layer_names)
    return (
        os.path.join(
            base_path,
            dataset + "_" + dtype + "_" + joined_layer_names + "_ats" + ".npy",
        ),
        os.path.join(base_path, dataset + "_" + dtype + "_pred" + ".npy"),
    )

def get_ats(
    model,
    dataset,
    name,
    layer_names,
    save_path=None,
    batch_size=128,
    is_classification=False,
    num_classes=None,
    num_proc=10,
):
    """Extract activation traces of dataset from model.

    Args:
        model (keras model): Subject model.
        dataset (list): Set of inputs fed into the model.
        name (str): Name of input set.
        layer_names (list): List of selected layer names.
        save_path (tuple): Paths of being saved ats and pred.
        batch_size (int): Size of batch when serving.
        is_classification (bool): Task type, True if classification task or False.
        num_classes (int): The number of classes (labels) in the dataset.
        num_proc (int): The number of processes for multiprocessing.

    Returns:
        ats (list): List of (layers, inputs, neuron outputs).
        pred (list): List of predicted classes or steering angles.
    """

    temp_model = Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output for layer_name in layer_names],
    )

    prefix = info("[" + name + "] ")
    p = Pool(num_proc)
    print(prefix + "Model serving")
    print("111111111111111111111111")
    if not is_classification:
        predictions = model.predict(dataset)
        pred = predictions.squeeze()  # 获取预测结果，并去除可能的多余维度
    else:
        predictions = model.predict(dataset)
        pred = np.argmax(predictions, axis=1)
    print("pred111222:",pred)

    if len(layer_names) == 1:
        layer_outputs = [
            temp_model.predict(dataset, batch_size=batch_size, verbose=1)
        ]
    else:
        layer_outputs = temp_model.predict(dataset, batch_size=batch_size, verbose=1)

    print(prefix + "Processing ATs")
    ats = None
    for layer_name, layer_output in zip(layer_names, layer_outputs):
        print("Layer: " + layer_name)
        if layer_output[0].ndim == 3:
            # For convolutional layers
            layer_matrix = np.array(
                p.map(_aggr_output, [layer_output[i] for i in range(len(dataset))])
            )
        else:
            layer_matrix = np.array(layer_output)

        if ats is None:
            ats = layer_matrix
        else:
            ats = np.append(ats, layer_matrix, axis=1)
            layer_matrix = None

    if save_path is not None:
        np.save(save_path[0], ats)
        np.save(save_path[1], pred)

    return ats, pred

def find_closest_at(at, train_ats):
    """The closest distance between subject AT and training ATs.

    Args:
        at (list): List of activation traces of an input.
        train_ats (list): List of activation traces in training set (filtered)

    Returns:
        dist (int): The closest distance.
        at (list): Training activation trace that has the closest distance.
    """
    print("111len(train_ats):",len(train_ats))
    print("at:",at)
    print("train_ats:",train_ats)
    dist = np.linalg.norm(at - train_ats, axis=1)
    return (min(dist), train_ats[np.argmin(dist)])


def _get_train_target_ats(model, x_train, x_target, target_name, layer_names, args):
    """Extract ats of train and target inputs. If there are saved files, then skip it.

    Args:
        model (keras model): Subject model.
        x_train (list): Set of training inputs.
        x_target (list): Set of target (test or adversarial) inputs.
        target_name (str): Name of target set.
        layer_names (list): List of selected layer names.
        args: keyboard args.

    Returns:
        train_ats (list): ats of train set.
        train_pred (list): pred of train set.
        target_ats (list): ats of target set.
        target_pred (list): pred of target set.
    """

    saved_train_path = _get_saved_path(args.save_path, args.d, "train", layer_names)
    if os.path.exists(saved_train_path[0]):
        print(infog("Found saved {} ATs, skip serving".format("train")))
        # In case train_ats is stored in a disk
        train_ats = np.load(saved_train_path[0])
        train_pred = np.load(saved_train_path[1])
    else:
        train_ats, train_pred = get_ats(
            model,
            x_train,
            "train",
            layer_names,
            num_classes=None,
            is_classification=args.is_classification,
            save_path=saved_train_path,
        )
        print(infog("train ATs is saved at " + saved_train_path[0]))

    saved_target_path = _get_saved_path(
        args.save_path, args.d, target_name, layer_names
    )
    if os.path.exists(saved_target_path[0]):
        print(infog("Found saved {} ATs, skip serving").format(target_name))
        # In case target_ats is stored in a disk
        target_ats = np.load(saved_target_path[0])
        target_pred = np.load(saved_target_path[1])
    else:
        target_ats, target_pred = get_ats(
            model,
            x_target,
            target_name,
            layer_names,
            num_classes=None,
            is_classification=args.is_classification,
            save_path=saved_target_path,
        )
        print(infog(target_name + " ATs is saved at " + saved_target_path[0]))

    return train_ats, train_pred, target_ats, target_pred

def fetch_dsa(model, x_train, x_target, target_name, layer_names, args):
    """Distance-based SA

            Args:
                model (keras model): Subject model.
                x_train (list): Set of training inputs.
                x_target (list): Set of target (test or adversarial) inputs.
                target_name (str): Name of target set.
                layer_names (list): List of selected layer names.
                args: keyboard args.

            Returns:
                dsa (list): List of dsa for each target input.
            """
    #assert args.is_classification == True

    prefix = info("[" + target_name + "] ")
    train_ats, train_pred, target_ats, target_pred = _get_train_target_ats(
        model, x_train, x_target, target_name, layer_names, args
    )

    print("train_pred:",train_pred)

    class_matrix = {}
    all_idx = []
    for i, label in enumerate(train_pred):
        #print("label:",label)  #值都是0
        label_str = "{:.6f}".format(float(label))
        #print("label_str1:", label_str)
        if label_str not in class_matrix:
            class_matrix[label_str] = []
            #print("label_str2:",label_str)
        class_matrix[label_str].append(i)
        all_idx.append(i)
    #print("all_idx:",all_idx)
    #print("class_matrix:",class_matrix)
    dsa = []

    print(prefix + "Fetching DSA")
    for i, at in enumerate(tqdm(target_ats)):
        label = target_pred[i]
        label_str = "{:.6f}".format(float(label))
        if label_str in class_matrix:
            print("find_closest_at111")
            a_dist, a_dot = find_closest_at(at, train_ats[class_matrix[label_str]])
            print("find_closest_at222")
            #print("set(all_idx) - set(class_matrix[label_str]):",list(set(all_idx) - set(class_matrix[label_str])))
            b_dist, _ = find_closest_at(
            a_dot, train_ats[list(set(all_idx) - set(class_matrix[label_str]))]
            )
            dsa.append(a_dist / b_dist)
        else:
            print("Label {} not found in class matrix".format(label_str))

    return dsa

def _get_kdes(train_ats, train_pred, class_matrix, args):
    """Kernel density estimation

    Args:
        train_ats (list): List of activation traces in training set.
        train_pred (list): List of prediction of train set.
        class_matrix (list): List of index of classes.
        args: Keyboard args.

    Returns:
        kdes (dict): Dictionary of kdes per label if classification task.
        removed_cols (list): List of removed columns by variance threshold.
    """

    removed_cols = []
    if not args.is_classification:
        unique_labels = np.unique(train_pred)
        for label in unique_labels:
            #print("Unique labels in train_pred:", unique_labels)
            label_str = "{:.6f}".format(float(label))
            #print("llll", label_str)
            #print("llll22", class_matrix[label_str])
            col_vectors = np.transpose(train_ats[class_matrix[label_str]])
            #print("col_vectors:",col_vectors)
            '''for i in range(col_vectors.shape[0]):
                #print("np.var(col_vectors[i]):",np.var(col_vectors[i]),"i:",i)
                if (
                        np.var(col_vectors[i]) < args.var_threshold
                        and i not in removed_cols
                ):
                    removed_cols.append(i)
        print("removed_cols",removed_cols)
        print("removed_cols",len(removed_cols))'''
        kdes = {}
        #i = 1
        for label in tqdm(unique_labels, desc="kde"):
            #if i < 500:
                label_str = "{:.6f}".format(float(label))
                refined_ats = np.transpose(train_ats[class_matrix[label_str]])
                #print("refined_ats shape:", refined_ats.shape)
                #print("refined_ats1:", refined_ats)
                refined_ats = np.squeeze(refined_ats)
                refined_ats = refined_ats.flatten()  # 将二维数组展平为一维数组
                #print("refined_ats2:", refined_ats)

                if refined_ats.shape[0] == 0:
                    print(
                        warn("ats were removed by threshold1 {}".format(args.var_threshold))
                    )
                    break
                kdes[label_str] = gaussian_kde(refined_ats)
                #print("kdes[label_str]:", kdes[label_str])
            #i = i+1

    else:
        col_vectors = np.transpose(train_ats)
        for i in range(col_vectors.shape[0]):
            if np.var(col_vectors[i]) < args.var_threshold:
                removed_cols.append(i)

        refined_ats = np.transpose(train_ats)
        refined_ats = np.delete(refined_ats, removed_cols, axis=0)
        if refined_ats.shape[0] == 0:
            print(warn("ats were removed by threshold {}".format(args.var_threshold)))
        kdes = [gaussian_kde(refined_ats)]

    print(infog("The number of removed columns: {}".format(len(removed_cols))))

    return kdes, removed_cols

def _get_lsa(kde, at, removed_cols):
    refined_at = np.delete(at, removed_cols, axis=0)
    print("refined_at.shape:",refined_at.shape)
    #print("-kde.logpdf(np.transpose(refined_at)):",-kde.logpdf(np.transpose(refined_at)))
    total_density = np.mean(-kde.logpdf(np.transpose(refined_at)))
    print("total_density:", total_density)
    return total_density

def fetch_lsa(model, x_train, x_target, target_name, layer_names, args):
    """Likelihood-based SA

    Args:
        model (keras model): Subject model.
        x_train (list): Set of training inputs.
        x_target (list): Set of target (test or[] adversarial) inputs.
        target_name (str): Name of target set.
        layer_names (list): List of selected layer names.
        args: Keyboard args.

    Returns:
        lsa (list): List of lsa for each target input.
    """

    prefix = info("[" + target_name + "] ")
    train_ats, train_pred, target_ats, target_pred = _get_train_target_ats(
        model, x_train, x_target, target_name, layer_names, args
    )

    class_matrix = {}
    class_matrix1 = {}
    if not args.is_classification:
        print("7777777777777")
        for label in train_pred:
            label_str = "{:.6f}".format(float(label))
            if label_str not in class_matrix:
                class_matrix[label_str] = []
            class_matrix[label_str].extend(np.where(train_pred == label)[0].tolist())

        for label in target_pred:
            label_str = "{:.6f}".format(float(label))
            #print("label_str7777777", label_str)
            if label_str not in class_matrix1:
                class_matrix1[label_str] = []
            class_matrix1[label_str].extend(np.where(target_pred == label)[0].tolist())

    #kdes, removed_cols = _get_kdes(train_ats, train_pred, class_matrix, args)
    kdes1, removed_cols1 = _get_kdes(target_ats, target_pred, class_matrix1, args)
    #print("kdes111111111111111:",kdes)
    #print("kdes111111111111111:",kdes1)

    lsa = []
    print(prefix + "Fetching LSA")
    if not args.is_classification:
        for i, at in enumerate(tqdm(target_ats[:20])):
            label = target_pred[i]
            label_str = "{:.6f}".format(float(label))
            #print("label_str:",label_str)
            kde = kdes1.get(label_str)
            if kde is not None:  # 检查kde是否为None
                print("6666666666666")
                print("kde1111111111111:", kde)
                print("at:",at)
                print("removed_cols1:",removed_cols1)
                lsa.append(_get_lsa(kde, at, removed_cols1))
            else:
                print(warn("KDE not found for label: {}".format(label_str)))
                lsa.append(None)
    else:
        kde = kdes1[0]
        for at in tqdm(target_ats):
            lsa.append(_get_lsa(kde, at, removed_cols1))

    return lsa

def get_sc(lower, upper, k, sa):
    """Surprise Coverage

    Args:
        lower (int): Lower bound.
        upper (int): Upper bound.
        k (int): The number of buckets.
        sa (list): List of lsa or dsa.

    Returns:
        cov (int): Surprise coverage.
    """

    buckets = np.digitize(sa, np.linspace(lower, upper, k))
    return len(list(set(buckets))) / float(k) * 100
