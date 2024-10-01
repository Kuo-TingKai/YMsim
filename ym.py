import numpy as np

# 4D SU(2) Yang-Mills 理論模擬
# 使用Monte Carlo方法
# 計算Wilson環/Polyakov環

def wilson_action(U):
    action = 0
    lattice_size = U.shape[1]
    for mu in range(4):
        for nu in range(mu+1, 4):
            for x in range(lattice_size):
                for y in range(lattice_size):
                    for z in range(lattice_size):
                        for t in range(lattice_size):
                            x_next = (x + 1) % lattice_size
                            y_next = (y + 1) % lattice_size if nu == 1 else y
                            z_next = (z + 1) % lattice_size if nu == 2 else z
                            t_next = (t + 1) % lattice_size if nu == 3 else t
                            
                            plaquette = np.trace(
                                U[mu,x,y,z,t] @
                                U[nu,x_next,y,z,t] @
                                U[mu,x_next,y_next,z_next,t_next].conj().T @
                                U[nu,x,y,z,t].conj().T
                            )
                            action += np.real(plaquette)
    return -action / lattice_size**4

def generate_su2_matrix():
    a = np.random.randn(3)
    a /= np.linalg.norm(a)
    theta = np.random.rand() * np.pi
    return np.cos(theta) * np.eye(2) + 1j * np.sin(theta) * np.sum([a[i] * pauli[i] for i in range(3)], axis=0)

pauli = [np.array([[0, 1], [1, 0]]), 
         np.array([[0, -1j], [1j, 0]]), 
         np.array([[1, 0], [0, -1]])]

def wilson_loop(U, r, t):
    lattice_size = U.shape[1]
    loop = np.eye(2)
    x, y, z, t0 = 0, 0, 0, 0
    for i in range(r):
        loop = loop @ U[0,x,y,z,(t0+i)%lattice_size]
    x = r % lattice_size
    for i in range(t):
        loop = loop @ U[3,x,y,z,(t0+i)%lattice_size]
    for i in range(r):
        loop = loop @ U[0,(x-i-1)%lattice_size,y,z,(t0+t)%lattice_size].conj().T
    for i in range(t):
        loop = loop @ U[3,0,y,z,(t0+t-i-1)%lattice_size].conj().T
    return np.trace(loop).real / 2

def polyakov_loop(U):
    lattice_size = U.shape[1]
    loop = np.eye(2)
    for t in range(lattice_size):
        loop = loop @ U[3,0,0,0,t]
    return np.trace(loop).real / 2

def main():
    lattice_size = 8  # 減小格點大小以加快計算
    num_iterations = 1000  # 減少迭代次數以加快測試
    beta = 2.3  # 反耦合常數

    U = np.array([[[[[generate_su2_matrix() for _ in range(lattice_size)] 
                     for _ in range(lattice_size)]
                    for _ in range(lattice_size)]
                   for _ in range(lattice_size)]
                  for _ in range(4)])

    print(f"U shape: {U.shape}")
    print(f"U[0,0,0,0,0] shape: {U[0,0,0,0,0].shape}")
    print(f"U[0,0,0,0,0] content:\n{U[0,0,0,0,0]}")

    initial_temp = 1.0
    final_temp = 0.1
    cooling_rate = (final_temp / initial_temp) ** (1 / num_iterations)
    
    temperature = initial_temp

    for i in range(num_iterations):
        mu = np.random.randint(4)
        x, y, z, t = [np.random.randint(lattice_size) for _ in range(4)]
        
        old_link = U[mu,x,y,z,t].copy()
        new_link = old_link @ generate_su2_matrix()
        
        U[mu,x,y,z,t] = new_link
        new_action = wilson_action(U)
        U[mu,x,y,z,t] = old_link
        old_action = wilson_action(U)
        
        delta_S = new_action - old_action
        
        if delta_S < 0 or np.random.rand() < np.exp(-beta * delta_S / temperature):
            U[mu,x,y,z,t] = new_link
        
        temperature *= cooling_rate
        
        if i % 100 == 0:  # 增加輸出頻率
            action = wilson_action(U)
            w_loop = wilson_loop(U, 2, 2)
            p_loop = polyakov_loop(U)
            print(f"迭代 {i}: 作用量 = {action}, Wilson環 = {w_loop}, Polyakov環 = {p_loop}, 溫度 = {temperature}")

    final_action = wilson_action(U)
    print(f"最終作用量: {final_action}, 最終溫度: {temperature}")
    return final_action

def run_multiple_simulations(num_runs):
    actions = []
    for run in range(num_runs):
        print(f"運行模擬 {run + 1}/{num_runs}")
        final_action = main()
        actions.append(final_action)
    
    mean_action = np.mean(actions)
    std_action = np.std(actions)
    
    print(f"平均作用量: {mean_action} ± {std_action}")

if __name__ == "__main__":
    run_multiple_simulations(3)  # 運行3次模擬