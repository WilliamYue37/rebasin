import matplotlib.pyplot as plt

def plot_interp_acc(lambdas, train_acc, test_acc):
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.plot(lambdas,
          train_acc,
          linestyle="solid",
          color="tab:blue",
          linewidth=2,
          label="Train, permuted interp.")
  ax.plot(lambdas,
          test_acc,
          linestyle="solid",
          color="tab:orange",
          linewidth=2,
          label="Test, permuted interp.")
  ax.set_xlabel("$\lambda$")
  ax.set_xticks([0, 1])
  ax.set_xticklabels(["Model $A$", "Model $B$"])
  ax.set_ylabel("Accuracy")
  # TODO label x=0 tick as \theta_1, and x=1 tick as \theta_2
  ax.set_title(f"Accuracy between the two models")
  ax.legend(loc="lower right", framealpha=0.5)
  fig.tight_layout()
  return fig
