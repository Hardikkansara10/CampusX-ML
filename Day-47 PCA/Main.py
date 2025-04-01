import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

# np.random.seed(23)

mu_vec1 = np.array([0, 0, 0])
cov_mat1 = np.identity(3)
class1_sample = np.random.multivariate_normal(mu_vec1, cov_mat1, 20)

df = pd.DataFrame(class1_sample, columns=['feature1', 'feature2', 'feature3'])
df['target'] = 1

mu_vec2 = np.array([1, 1, 1])
cov_mat2 = np.identity(3)
class2_sample = np.random.multivariate_normal(mu_vec2, cov_mat2, 20)

df1 = pd.DataFrame(class2_sample, columns=['feature1', 'feature2', 'feature3'])
df1['target'] = 0

df = pd.concat([df, df1], ignore_index=True).sample(frac=1).reset_index(drop=True)

fig = px.scatter_3d(df, x='feature1', y='feature2', z='feature3', color=df['target'].astype(str))
fig.update_traces(marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')))
fig.show()

scaler = StandardScaler()
df.iloc[:, 0:3] = scaler.fit_transform(df.iloc[:, 0:3])

covariance_matrix = np.cov(df.iloc[:, 0:3].T)
print('Covariance Matrix:\n', covariance_matrix)

eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
print("Eigenvalues:", eigen_values)
print("Eigenvectors:\n", eigen_vectors)

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs
    
    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['feature1'], df['feature2'], df['feature3'], color='blue', alpha=0.2)
ax.scatter([df['feature1'].mean()], [df['feature2'].mean()], [df['feature3'].mean()], color='red', s=100)

for v in eigen_vectors.T:
    arrow = Arrow3D([df['feature1'].mean(), v[0]], 
                    [df['feature2'].mean(), v[1]], 
                    [df['feature3'].mean(), v[2]], 
                    mutation_scale=20, lw=3, arrowstyle="-|>", color="r")
    ax.add_artist(arrow)

ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
plt.title('Eigenvectors')
plt.show()

pc = eigen_vectors[:, :2]  
transformed_df = np.dot(df.iloc[:, 0:3], pc)
new_df = pd.DataFrame(transformed_df, columns=['PC1', 'PC2'])
new_df['target'] = df['target'].astype(str)

fig = px.scatter(new_df, x='PC1', y='PC2', color='target', color_discrete_sequence=px.colors.qualitative.G10)
fig.update_traces(marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')))
fig.show()