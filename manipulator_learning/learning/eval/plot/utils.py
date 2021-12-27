import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def setup_long_fig(num_subfigs, y_label, x_label, font_size, width=6.4, height=4.8):
    fig = plt.figure(figsize=[num_subfigs * width, height])
    ax = fig.add_subplot(111)
    ax.set_ylabel(y_label, fontsize=font_size)
    ax.set_xlabel(x_label, fontsize=font_size)
    remove_axis_lines_ticks(ax)
    return fig


def remove_axis_lines_ticks(ax):
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)


def get_suc_rew_figs(data_dirs, x_label, font_size):
    # avg_suc_fig = plt.figure(0, figsize=plt.figaspect(1 / len(data_dirs)))
    avg_suc_fig = plt.figure(0, figsize=[len(data_dirs) * 6.4, 4.8])
    avg_suc_fig_ax = avg_suc_fig.add_subplot(111)
    avg_suc_fig_ax.set_ylabel('Success Rate', fontsize=font_size)
    avg_suc_fig_ax.set_xlabel(x_label, fontsize=font_size)
    remove_axis_lines_ticks(avg_suc_fig_ax)
    # plt.suptitle('Average Success')
    avg_rew_fig = plt.figure(1, figsize=plt.figaspect(1 / len(data_dirs)))
    avg_rew_fig_ax = avg_rew_fig.add_subplot(111)
    avg_rew_fig_ax.set_ylabel('Average Reward (Scaled)', fontsize=font_size)
    avg_rew_fig_ax.set_xlabel(x_label, fontsize=font_size)
    remove_axis_lines_ticks(avg_rew_fig_ax)
    # plt.suptitle('Average Reward (Scaled)')
    avg_rew_all_fig = plt.figure(2, figsize=plt.figaspect(1 / len(data_dirs)))
    avg_rew_all_fig_ax = avg_rew_all_fig.add_subplot(111)
    avg_rew_all_fig_ax.set_ylabel('Average Reward (Scaled)', fontsize=font_size)
    avg_rew_all_fig_ax.set_xlabel(x_label, fontsize=font_size)
    remove_axis_lines_ticks(avg_rew_all_fig_ax)
    # plt.suptitle('Average Reward (Scaled)')
    return avg_suc_fig, avg_rew_fig, avg_rew_all_fig


def setup_pretty_plotting():
    # pretty plotting stuff
    font_params = {
        "font.family": "serif",
        "font.serif": "Times",
        "text.usetex": True,
        "pgf.rcfonts": False
    }
    plt.rcParams.update(font_params)


def final_ax_formatting(ax, i_dir, font_size):
    # formatting
    if i_dir == 0:
        ax.legend(fontsize=font_size - 6)

    ax.tick_params(axis='both', which='minor')
    ax.xaxis.set_tick_params(labelsize=font_size - 8)
    ax.yaxis.set_tick_params(labelsize=font_size - 8)
    # if i_dir > 0:
    #   ax.set_yticklabels([])


def get_max_values_of_conds(stats_dict):
    # for ROC
    all_max_consec_non_inc_q_neg_ds = []
    all_max_consec_neg_ds = []
    for c, ckpt in enumerate(stats_dict['total_numsteps']):
        d_q_shape = stats_dict['d_values'][c].shape  # num seeds x num eps x ep length
        d_values = stats_dict['d_values'][c].reshape([d_q_shape[0] * d_q_shape[1], d_q_shape[2]])
        q_values = stats_dict['q_values'][c].reshape([d_q_shape[0] * d_q_shape[1], d_q_shape[2]])
        max_consec_non_inc_q_neg_ds = []
        max_consec_neg_ds = []
        for ep in range(d_q_shape[0] * d_q_shape[1]):
            max_consec_non_inc_q_neg_d = 0
            max_consec_neg_d = 0
            consec_non_inc_q_neg_d = 0
            consec_neg_d = 0
            last_q = -1e10
            for t in range(d_q_shape[2]):
                if d_values[ep, t] < 0:
                    # if d_values[ep, t] < .05:
                    consec_neg_d += 1
                    max_consec_neg_d = max(consec_neg_d, max_consec_neg_d)
                    if q_values[ep, t] - last_q < 0:
                        consec_non_inc_q_neg_d += 1
                        max_consec_non_inc_q_neg_d = max(consec_non_inc_q_neg_d, max_consec_non_inc_q_neg_d)
                    else:
                        consec_non_inc_q_neg_d = 0
                else:
                    consec_neg_d = 0
                    consec_non_inc_q_neg_d = 0
                last_q = q_values[ep, t]
            max_consec_non_inc_q_neg_ds.append(max_consec_non_inc_q_neg_d)
            max_consec_neg_ds.append(max_consec_neg_d)
        all_max_consec_non_inc_q_neg_ds.append(max_consec_non_inc_q_neg_ds)
        all_max_consec_neg_ds.append(max_consec_neg_ds)
    return np.array(all_max_consec_non_inc_q_neg_ds), np.array(all_max_consec_neg_ds)


def get_max_consec_doubt(doubts, thresh):
    max_consec_doubts = []
    for ep in range(doubts.shape[0]):
        max_consec_doubt = 0
        consec_doubt = 0
        for t in range(doubts.shape[1]):
            if doubts[ep, t] > thresh:
                consec_doubt += 1
                max_consec_doubt = max(consec_doubt, max_consec_doubt)
            else:
                consec_doubt = 0
        max_consec_doubts.append(max_consec_doubt)
    return np.array(max_consec_doubts)


def get_confusion_matrix(stats_dict):
    recall = [];
    precision = [];
    accuracy = [];
    fail_ts = []
    for c, ckpt in enumerate(stats_dict['total_numsteps']):
        alpha = stats_dict['alpha'][c]
        beta = stats_dict['beta'][c]
        # alpha = 5
        # beta = 20
        success = stats_dict['successes'][c].flatten()
        d_q_shape = stats_dict['d_values'][c].shape  # num seeds x num eps x ep length
        d_values = stats_dict['d_values'][c].reshape([d_q_shape[0] * d_q_shape[1], d_q_shape[2]])
        q_values = stats_dict['q_values'][c].reshape([d_q_shape[0] * d_q_shape[1], d_q_shape[2]])
        rewards = stats_dict['rewards'][c].flatten()
        tp = 0;
        fp = 0;
        fn = 0;
        tn = 0;
        ckpt_fail_t = []
        for ep in range(success.shape[0]):
            consec_non_inc_q_neg_d = 0
            consec_neg_d = 0
            last_q = -1e10
            beta_cond = False
            alpha_cond = False
            fail_t = 0
            for t in range(d_q_shape[2]):
                if d_values[ep, t] < 0:
                    consec_neg_d += 1
                    if q_values[ep, t] - last_q < 0:
                        consec_non_inc_q_neg_d += 1
                    else:
                        consec_non_inc_q_neg_d = 0
                else:
                    consec_neg_d = 0
                    consec_non_inc_q_neg_d = 0
                last_q = q_values[ep, t]

                if consec_neg_d >= beta or consec_non_inc_q_neg_d >= alpha:
                    beta_cond = consec_neg_d >= beta
                    alpha_cond = consec_non_inc_q_neg_d >= alpha
                    fail_t = t
                    break
            if alpha_cond or beta_cond:
                pred_fail = True
            else:
                pred_fail = False

            if not success[ep] and pred_fail:
                tp += 1
            elif not success[ep] and not pred_fail:
                fn += 1
            elif success[ep] and pred_fail:
                fp += 1
            elif success[ep] and not pred_fail:
                tn += 1
            ckpt_fail_t.append(fail_t)

        if (tp + fn) == 0:
            recall.append(1)
        else:
            recall.append(tp / (tp + fn))
        if (tp + fp) == 0:
            precision.append(1)
        else:
            precision.append(tp / (tp + fp))
        accuracy.append((tp + tn) / (tp + fn + fp + tn))
        fail_ts.append(ckpt_fail_t)
    return recall, precision, accuracy, fail_ts


def get_fpr_tpr(pred_fail, successes):
    tp = (pred_fail & np.invert(successes)).sum()
    fn = (np.invert(pred_fail) & np.invert(successes)).sum()
    fp = (pred_fail & successes).sum()
    tn = (np.invert(pred_fail) & successes).sum()
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    return fpr, tpr


def get_roc_stats_d(successes, d_values, test_values):
    roc_points = []
    for param in test_values:
        pred_fail = param < d_values
        roc_points.append(get_fpr_tpr(pred_fail, successes))
    return np.array(roc_points)


def get_roc_stats_ensemble(successes, doubts, test_values):
    roc_points = []
    for param in test_values:
        pred_fail = doubts >= param
        roc_points.append(get_fpr_tpr(pred_fail, successes))
    return np.array(roc_points)


def get_roc_stats_ensemble_fixedfire(successes, doubts, fire_pred_fail, test_values):
    roc_points = []
    for param in test_values:
        pred_fail = np.any([doubts >= param, fire_pred_fail], axis=0)
        roc_points.append(get_fpr_tpr(pred_fail, successes))
    return np.array(roc_points)


def get_roc_stats_fixedensemble_fire(successes, doubt_pred_fail, fire_pred_fail_all, test_values):
    roc_points = []
    for param in test_values:
        pred_fail = np.any([doubt_pred_fail, fire_pred_fail_all[param]], axis=0)
        roc_points.append(get_fpr_tpr(pred_fail, successes))
    return np.array(roc_points)


def get_roc_stats(successes, max_param_value, test_range):
    roc_points = []
    for param in range(test_range):  # diff values of params, should be length of episodes
        pred_fail = param <= max_param_value
        roc_points.append(get_fpr_tpr(pred_fail, successes))
    return np.array(roc_points)


def get_roc_stats_multi_param(successes, max_param_values, test_combos):
    roc_points = []
    for params in test_combos:  # diff values of params, should be length of episodes
        pred_fail = ((params[0] <= max_param_values[0]) | (params[1] <= max_param_values[1]))
        roc_points.append(get_fpr_tpr(pred_fail, successes))
    return np.array(roc_points)


def get_roc_stats_qual(successes, qual_values, test_values, qual_should_be_lt_param=True):
    roc_points = []
    for param in test_values:
        if qual_should_be_lt_param:
            pred_fail = param >= qual_values
        else:
            pred_fail = param < qual_values
        roc_points.append(get_fpr_tpr(pred_fail, successes))
    return np.array(roc_points)


def get_stats_dict(np_filename):
    stats_dict = np.load(np_filename)
    return {key: stats_dict[key] for key in stats_dict.files}
