import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-5,5,100)
y1 = (1.0/np.log(2))*np.log(1+np.exp(-1*x))
hinge_loss =  np.maximum(1-x,0)
plt.plot(x,y1,label='$g(\\zeta)$')
plt.plot(x,hinge_loss, label='Hinge Loss')
plt.xlim([-2,5])
plt.ylim([0,3])
plt.xlabel('$\\zeta$')
plt.ylabel('Loss')
plt.legend(loc="upper right")
plt.title("Hinge Loss vs. $g(\\zeta)$")
plt.show()
