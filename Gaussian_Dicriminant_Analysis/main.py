import numpy as np
import gda
import plot_graph
import generate

xs,ys,meu0,meu1,_,_=generate.generate_binary_2d_dataset(100)
params_plot,params_predict=gda.gda_binary_learn(xs,ys)
print(gda.gda_binary_predict(np.array([1,5]),params_predict))
plot_graph.plot_gda_level_curves(xs,ys,params_plot)

print(f"MEU_0:Actual{meu0} ,Calculated:{params_predict[1]} ")
print(f"MEU_0:Actual{meu1} ,Calculated:{params_predict[2]} ")

