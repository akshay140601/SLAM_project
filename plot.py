import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class AnimatedMap:
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.initialized = False

    def update_scatter_plot(self, x_new, y_new, z_new):
        if not self.initialized:
            self.initialized = True
            self.ax.scatter(x_new, y_new, z_new, c='red')
        else:
            self.ax.scatter(x_new, y_new, z_new, c='red')

        plt.pause(5)

'''if __name__ == '__main__':

    initial_x_values = [1, 2, 3, 4, 5]
    initial_y_values = [2, 3, 5, 7, 11]
    initial_z_values = [0, 1, 2, 3, 4]

    animated_map = AnimatedMap()
    animated_map.update_scatter_plot(initial_x_values, initial_y_values, initial_z_values)
    new_x_values = [6, 7, 8]
    new_y_values = [13, 17, 19]
    new_z_values = [5, 6, 7]
    animated_map.update_scatter_plot(new_x_values, new_y_values, new_z_values)

    plt.show()'''
