import numpy as np
import matplotlib.pyplot as plt

def fig_colored_matrix(
	M,
	ax=None,
	xticks=None,
	yticks=None,
	show_colorbar=False,
	figsize=None,
	vmin=0,
	vmax=1
	):

	if ax:
		plt.sca(ax)
	else:
		if not figsize:
			nx = M.shape[0]
			ny = M.shape[1]
			figsize = (nx,ny*3.0/4.0)
		fig = plt.figure(figsize=figsize)
		ax = plt.axes()

	if vmin is None:
		vmin = np.min(M)
	if vmax is None:
		vmax = np.max(M)
	plt.imshow(M,vmin=vmin,vmax=vmax)
	
	for i in range(M.shape[0]):
		for j in range(M.shape[1]):
			if M[j,i] >= 0.5*(vmax-vmin):
				plt.text(i,j,f"{M[j,i]:.0f}",
						color="k",
						weight="bold",
						ha="center",
						va="center")
			else:
				plt.text(i,j,f"{M[j,i]:.0f}",
						color="w",
						weight="bold",
						ha="center",
						va="center")
	
	if not xticks:
		xticks = np.arange(M.shape[0])
	if not yticks:
		yticks = np.arange(M.shape[1])

	plt.xticks(range(M.shape[0]),xticks)
	plt.yticks(range(M.shape[1]),yticks)

	if show_colorbar:
		plt.colorbar()

	return ax