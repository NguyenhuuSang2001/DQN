from lib import *

class IoT_Semantic_Wireless_Harvesting():
    def __init__(self, N_IoT=3, num_channel=2, sensing_rate=2, t_harv=[0,2,4], num_level_p_trans=2, max_p_trans=7, max_step=20):

        # print(f"N_IoT: {N_IoT}, N_channels: {num_channel}, sensing_rate: {sensing_rate}\n")
        
        self.num_IoT = N_IoT
        self.num_channel = num_channel
        self.sensing_rate = sensing_rate
        self.T_harv = t_harv
        self.num_level_p_trans = num_level_p_trans
        self.max_p_trans = max_p_trans
        self.max_step = max_step

        self.confidence_values = np.array([ 0.1])
        # self.comp_capability = np.array([2.4 * 10**9] * self.num_IoT)
    ##    
        self.h_i = np.array([random.randint(200, 500)/1e8 for _ in range(self.num_IoT)])
        self.d_i = np.array([random.randint(20, 30) for _ in range(self.num_IoT)])
        self.E_i0 = np.array([random.randint(1, 2) for _ in range(self.num_IoT)])
        self.t_sens = np.array([1]*self.num_IoT)

        # e_sen: 2->5, e_comp:  
        self.E_capacity = np.array([random.randint(7, 10)]*self.num_IoT) # Capacity energy UAV

        # self.t_harv = np.array([2]*self.num_IoT)

        self.N_img = self.sensing_rate*self.t_sens[0]

        self.L = 1024*1096 #bits
        # L = 8*(1024**2) #bits
        self.size_img_defaul = np.array([random.randint(500, 1000)*1e-3*self.L for _ in range(self.num_IoT)])
        self.kappa = 10**-28


        self.start = True
        self.e_0_default = np.array([random.randint(3, 10) for _ in range(self.num_IoT)])
        self.initialization()
        self.action_space = ActionSpace(self._size_action_space())

    def initialization(self):
        # self.seed(1)
        self.count_step = 0
        # Initialize the environment
        
        self.D_i = self._get_Di()
        self.e_i = self._get_ei()
        self.f_i = self._get_fi()
        self.state = np.concatenate((self.e_i, self.f_i, self.D_i))

        self.observation_space = np.array([0 for _ in range(self.state_size())])

    def _size_action_space(self):
        self._space_t_harv = len(self.T_harv)**self.num_IoT
        self._space_delta = 2**self.num_IoT #option 0 or 1
        self._space_gamma = len(self.confidence_values)**self.num_IoT 
        self._space_alpha = 2**self.num_IoT #option 0 or 1
        self._space_p_trans = (self.num_level_p_trans+1)**self.num_IoT #add option 0

        return self._space_t_harv*self._space_delta*self._space_gamma * self._space_alpha * self._space_p_trans

    def state_size(self):
        return len(self.state)

    def state_shape(self):
        return (len(self.state), )
    
    def reset(self):
        self.start = True
        self.initialization()

        #---
        # self.state = np.array([random.random() for _ in range(self.num_IoT*4)])

        return self.state, {}


    def step(self, action):
        done = False
        self.count_step += 1

        # Simulate the communication process and calculate reward
        reward = self._calculate_reward(action)
        # print("reward: ", reward)
        # print("."*42)
        if reward < 0 or self.count_step >= self.max_step:
            done = True
            self.count_step = 0

        # Simulate state transition
        self.state = self._state_next()
        
        #---
        # self.state = np.array([random.random() for _ in range(self.num_IoT*4)])

        truncated, terminated = done, done
        
        return self.state, reward, terminated, truncated, {}
    
    def convert_action(self, action_index):
        # t_harv, delta, gamma, alpha, p_trans
       
        ind = action_index

        t_harv = ind%self._space_t_harv
        ind = ind // self._space_t_harv
        # print("t_harv", t_harv)

        delta = ind%self._space_delta
        ind = ind // self._space_delta

        gamma = ind%self._space_gamma
        ind = ind // self._space_gamma

        alpha = ind%self._space_alpha
        ind = ind // self._space_alpha

        p_trans = ind

        # print(t_harv, delta, gamma, alpha, p_trans)

        
        list_t_harv = convert_base(t_harv, len(self.T_harv), self.num_IoT )
        list_delta = convert_base(delta, 2, self.num_IoT )
        list_gamma = convert_base(gamma, len(self.confidence_values), self.num_IoT)
        list_alpha = convert_base(alpha, 2, self.num_IoT )
        list_p_trans = convert_base(p_trans, self.num_level_p_trans + 1, self.num_IoT)

        action = list_p_trans
        action.extend(list_alpha)
        action.extend(list_gamma)
        action.extend(list_delta)
        action.extend(list_t_harv)

        return np.array(action)

# Ham tinh state
    def _state_next(self):

        self.e_i = self._get_ei()

        self.f_i = self._get_fi()

        self.D_i = self._get_Di()
        
        self.state = np.concatenate((self.e_i, self.f_i, self.D_i))

        return self.state

    def _get_ei(self): 
        if self.start == False:
            self.e_0 = self._calculate_e_0()
        else:
            self.e_0 = self.e_0_default
            self.start = False
            
        print_array("e_0:", self.e_0)

        self.size_img = self.D_i*1e6
        self.e_sens = self._calculate_e_sens()

        e_i = self.e_0 - self.e_sens
        print_array("e_i   :", e_i)
        # print("="*42)

        return e_i


    def _get_fi(self):
        self.f_mean = 2.4*10**9
        self.f_variance = 0.05*self.f_mean

        f_i = np.array([int(np.random.normal(self.f_mean, self.f_variance)) for _ in range(self.num_IoT)])
        
        #---
        f_i = f_i / self.f_mean
        print_array("f_i:", f_i)
        return f_i

    def _calculate_denta_t(self):
        T = np.array([10]*self.num_IoT)
        print_array("t_harv:",self.t_harv)
        print_array("t_trans:", self.t_trans)
        self.t_comp =  self._calculate_t_comp()

        denta_t = T - (self.t_harv + self.delta*(self.t_comp + self.t_select) - self.t_trans)
        # print("-"*42)
        print_array("denta_t", denta_t)
        # print("="*42)
        
        return denta_t
    
    def _get_Di(self):
        D_i = (1 + np.array([random.randint(-500, 500)*1e-3 for _ in range(self.num_IoT)]))*self.L
        D_i = D_i/1e6
        print_array("D_i:", D_i)
        return D_i
    
    def _calculate_G_i(self):
        G_i = np.array([0]*self.num_IoT)
        c_i = 5e-3*self.size_img
        print_array("size_img:", self.size_img)
        for i in range(self.num_IoT):
            for j in range(self.N_img):
                G_i[i] += (1+random.randint(10, 100)*1e-4)*c_i[i]
        #---
        G_i = G_i/1e3
        
        return G_i

    def seed(self, random_seed):
        np.random.seed(random_seed)
        random.seed(random_seed)

# Ham phu thuoc e
    def _calculate_e_0(self):
        e = np.column_stack([self.e_i + self.e_harv, self.E_capacity])
        e = np.min(e, axis=1)
        # print("-"*42)
        print_array("e_after_h:", e)


        e_0 = e - self.delta * (self.e_comp + self.e_select) - self.e_trans

        return e_0
    
    def _calculate_e_sens(self): 
        #2 -> 5 J

        e_consumed_sense_bit = 50*10**-9

        e_sens = e_consumed_sense_bit * self.N_img * self.size_img 

        print_array("e_sens:", e_sens)
        return e_sens

    def _calculate_e_comp(self):
        r_CPU = 3000*self.size_img*self.N_img

        e_comp = self.kappa * r_CPU * ((self.f_i*self.f_mean)**2)

        print_array("e_comp:", e_comp)

        return e_comp

    def _calculate_e_harv(self):
        P_harv = self._calculate_P_harv()

        e_harv = P_harv * self.t_harv
        print_array("e_harv:", e_harv)
        return e_harv

    def _calculate_e_select(self):
        C_j1 = np.array([0.3]*self.num_IoT)
        C_j2 = np.array([0.05]*self.num_IoT)
        C_j3 = 0.2

        r_CPU = C_j1*(self.rho - C_j2)**C_j3

        e_select = self.kappa * r_CPU * ((self.f_i*self.f_mean)**2)
        print_array("e_select:", e_select)
        return e_select

    def _calculate_e_trans(self):
        e_trans = self.p_trans * self.t_trans
        print_array("e_trans :", e_trans)
        return e_trans

    def _calculate_P_harv(self):
        xi = 0.8
        P = 10 #40 dBm -> 10 W
        alpha = 0.1

        h_i = np.array([random.randint(200, 500)/1e3 for _ in range(self.num_IoT)])
        tmp = (-1)**random.randint(1,2)*min(random.random(), 5e-4) 
        P_harv = xi * P * (h_i + tmp)* self.d_i**(-alpha)

        return P_harv


#Ham phu thuoc t
    def _calculate_R_i(self):
        W = 100*10**3
        sigma_2 = 10**-17

        # print_array("p_trans: ", self.p_trans)
        tmp = (-1)**random.randint(1,2)*min(random.random(), 5e-8) 
        # print("h_i: ", self.h_i)
        print_array("d_i: ", self.d_i)
        R_i = W*np.log2(1+(self.p_trans*(self.h_i + tmp))/(W*sigma_2*self.d_i**2))
        print_array("R_i", R_i)
        return R_i

# ------
    def _calculate_t_trans(self):
        self.Q_i = (1e6*self.D_i)**(1-self.delta)*(1e3*self.G_i*self.rho)**self.delta
        print_array("G_i:", self.G_i)
        print_array("rho:", self.rho)
        print_array("Q_i:", self.Q_i)
        R_i = self._calculate_R_i()

    # ---- fix div 0
        mask = R_i==0
        R_i[mask] = 10**20

        t_trans = self.Q_i / R_i
        

        print_array("t_trans: ", t_trans)
        return t_trans

# ------
    def _calculate_t_comp(self):
        # t_comp = np.array([random.randint(3*1e2, 6*1e2)*1e-2 for _ in range(self.num_IoT)])
        r_CPU = 3000*self.size_img*self.N_img

        t_comp = r_CPU/(self.f_i*self.f_mean)
        print_array("t_comp:", t_comp)
        return t_comp
    
    def _calculate_t_select(self):
        # t_select = 1e-2*np.array([random.random() for _ in range(self.num_IoT)])
        C_j1 = np.array([0.3]*self.num_IoT)
        C_j2 = np.array([0.05]*self.num_IoT)
        C_j3 = 0.2

        r_CPU = C_j1*(self.rho - C_j2)**C_j3
        t_select = r_CPU/(self.f_i*self.f_mean)
        print_array("t_select:", t_select)
        return t_select


#Ham tinh reward
    def _calculate_reward(self, action_index):
        # print("-"*42)
        # print_array("e_i", self.e_i)
        # print_array("D_i", self.D_i)
        # print_array("f_i", self.f_i)
        
        # t_harv, delta, gamma, alpha, p_trans
        action = self.convert_action(action_index)
        # print(action_index)

        action = action.reshape(-1, self.num_IoT)

        self.t_harv = action[0]
        print_array("t_harv:", self.t_harv)

        self.delta = action[1]
        print_array("delta:", self.delta)

        self.gamma = action[2]
        print_array("gamma:", self.gamma)
        
        self.alpha = action[3]
        print_array("alpha:", self.alpha)

        p = action[4]
        self.p_trans = self.max_p_trans/self.num_level_p_trans * p 
        print_array("p_trans:", self.p_trans)
        # print("+"*42)


        self.rho = np.array([1-self.confidence_values[i] for i in self.gamma])

        print_array("e_i:", self.e_i)
        # print_array("denta_t:", self.denta_t)

        self.e_harv = self._calculate_e_harv()
        self.G_i = self._calculate_G_i()

        self.t_comp = self.delta*self._calculate_t_comp()
        self.e_comp = self.delta*self._calculate_e_comp()
                
        self.t_select = self.delta*self._calculate_t_select()
        self.e_select = self.delta*self._calculate_e_select()

        self.t_trans = self._calculate_t_trans()
        self.e_trans = self._calculate_e_trans()

        self.denta_t = self._calculate_denta_t()

        e_0  = self._calculate_e_0()

        k0 = 2
        k1 = 3

        if np.sum(self.denta_t < 0) > 0 or np.sum(e_0 < self.E_i0) > 0 or np.sum(self.alpha) > self.num_channel or np.sum(self.alpha) == 0:
            return -2*k0
        
        reward = 1e-1*k0*np.sum(self.alpha*self.N_img*(self.D_i*1e2)**(1-self.delta)*(1 - np.exp(-k1*self.rho))**self.delta)
        
        return reward
    def render(self):
        print_array("e_i:", self.e_i)
        print_array("f_i:", self.f_i)
        print_array("G_i:", self.G_i)
        print_array("denta_t:", self.denta_t)

def env_example(N_IoT=2, num_channel=1, sensing_rate=3):
    # Example usage

    env = IoT_Semantic_Wireless_Harvesting(N_IoT=N_IoT,num_channel=num_channel, sensing_rate=sensing_rate)

    return env


def e_greedy(env):
    r_best = 0
    ind_r = 0
    for action in range(env.action_space.n):
        # print(env.state)
        reward = env._calculate_reward(action)
        if reward > r_best:
            r_best = reward
            ind_r = action
    # print("reward best:" ,r_best)
    return ind_r

if __name__ == "__main__":

 
    # Reset the environment
    env = env_example(N_IoT=2, num_channel=1, sensing_rate=3)
    initial_state = env.state
    # print("Initial State:")
    # env.render()
    # print("action space: ",env.action_space.n)

    action = random.randint(0, env.action_space.n)
    # action = env.action_space.n - 1
    for action in range(env.action_space.n):
        # print(env.state)
        env.reset()
        reward = env._calculate_reward(action)
        print("index action:  {} => {}".format(reward, env.convert_action(action)))

    # print(e_greedy(env))

    # print("index action: {} => {}".format(action, env.convert_action(action)))
    new_state, reward, done, _, info = env.step(action)
    # print("New State:")
    env.render()
    print("Reward:", reward)