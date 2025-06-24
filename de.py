class Indivi:
    # コンストラクタ
    def __init__(self, gene = None,EvalFunc = None):
        self.gene = gene
        self.EvalFunc = EvalFunc
        self.fitness = self.Evaluate()
        self.F  = INITIAL_F
        self.CR = INITIAL_CR

    # 評価モジュール
    def Evaluate(self):
        temp = (self.gene)
        fitness = self.EvalFunc(temp)
        return float(fitness)

    # 終了モジュール
    def Finish(self, best_gene):
        print(best_gene)
        out = self.EvalFunc(best_gene)
        print(out)



class adaptive_DE:
    # コンストラクタ
    def __init__(self, Z_test,EvalFunc,numofpopu,Population,seed_random,seed_np):
        self.EvalFunc = EvalFunc
        self.POPULATION = numofpopu
        self.seed_random = seed_random              # シード値：初期個体集団
        self.seed_np = seed_np
        random.seed(self.seed_random)           # 個体シャッフルで使用するシード値
        np.random.seed(self.seed_np)            # 差分ベクトル計算で使用するシード値

        random.shuffle(Z_test)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)
        for p in range(self.POPULATION):
            Population.append(Indivi(gene=Z_test[p],EvalFunc=self.EvalFunc))
        self.before = []

    # 突然変異
    def Mutation(self, Num_UpdatedIndividual):
        update = 0

        for i in range(self.POPULATION):        # 親個体を選択（親個体のインデックス）
            vector_xnew = [0 for i in range(DIMENSION)]
            p = np.arange(self.POPULATION);     # 0からPopulationまでの等差数列
            p = np.delete(p,i)                  # 現在の親個体を選択個体から除外
            np.random.shuffle(p)

            # jDE Algorithm. Decide parameters F and CR
            if (np.random.rand() < F_GAMMA):
                Population[i].F  = F_LOWEST + np.random.rand()*F_UPPER

            if (np.random.rand() < CR_GAMMA):
                Population[i].CR = np.random.rand()

            #-----------------------Mutation Operation----------------------
            vector_xx = Population[p[0]].gene + Population[i].F * (Population[p[2]].gene - Population[p[1]].gene)

            #-----------------------Crossover Operation----------------------
            jrand = np.random.randint(DIMENSION)
            for j in range(DIMENSION):
                if (np.random.rand() >  1.0-Population[i].CR and j!=jrand):
                    vector_xnew[j] = Population[i].gene[j]    # 親個体の要素引き継ぎ
                else:
                  if (vector_xx[j] >= MIN_value[j] and vector_xx[j] <= MAX_value[j]):    # 潜在表現が定義域内に収まる場合
                      vector_xnew[j] = vector_xx[j]
                  else:
                      vector_xnew[j] = Population[i].gene[j]

            vector_xnew = torch.tensor(vector_xnew,dtype=torch.float32)
            new_Individual = Indivi(vector_xnew,self.EvalFunc)

            #Set new parameters F and CR
            new_Individual.F = Population[i].F
            new_Individual.CR = Population[i].CR

            #-----------------------Select Operation-----------------------
            fitness_p = Population[i].Evaluate()
            Population[i].fitness = fitness_p
            if (new_Individual.fitness > fitness_p):
                Population[i] = new_Individual
                update += 1

        print("UpdatedIndividual=",update)
        Num_UpdatedIndividual.append(update)

    # 平均評価値の計算
    def Average_Fitness(self):
        x =0.0
        for i in range(self.POPULATION):
            x += Population[i].fitness
        string = "Average Fitness = " + str(x/self.POPULATION)
        print(string)
        return float(x/self.POPULATION)

    # 最良個体の選択（上位n体を選択するよう修正）
    def Best_Individual(self, generation, GENERATION, top_n=3):
        Population.sort(key=operator.attrgetter('fitness'), reverse=True)
        if generation == GENERATION - 1:
            print("----------------Best Individuals----------------")
            for i in range(min(top_n, len(Population))):
                print(f"Rank {i+1}:")
                print("Weight Sharing:", Population[i].fitness, "\n", Population[i].gene)

    # 最良個体のプロットモジュール（上位n体を選択するよう修正）
    def Best_plot(self, Bestfitness, Bestgene, top_n=3):
        # 上位n体の評価値と遺伝子を格納
        for i in range(min(top_n, len(Population))):
            Bestfitness.append(Population[i].fitness)     # i番目に良い個体の評価値格納
            Bestgene.append(Population[i].gene)           # i番目に良い個体の潜在表現格納

if __name__ == "__main__":

    # 潜在表現の取得
    with open(DRIVE_DIR + SPECIFIC_LATENT,'rb') as file:
        _, _, Z_test, Y_test = pickle.load(file)

    Z_test_tensor = torch.tensor(Z_test, dtype=torch.float32)


    # 各trial毎の最良個体リスト
    Average_fitness_trial = []
    Bestgene_trials = []
    Bestfitness_trials = []

    # パラメータの定義
    Population = []
    Average_fitness = []
    Bestfitness = []
    Bestgene = []
    Num_UpdatedIndividual = []

    # 学習済み評価関数（MLP）の読み込み
    EvalFunc = MLP_hid(INPUT_DIMENSION,HIDDEN_DIMENSION,OUTPUT_DIMENSION)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    EvalFunc_saved_name = MLP_PATH + "{0}_model_lr{1}_mse{2}_kld{3}_reg{4}.pth".format(TIMING,LR_MLP,MSE_ALPHA,KLD_BETA,REG_GAMMA)
    print(EvalFunc_saved_name)
    EvalFunc.load_state_dict(torch.load(EvalFunc_saved_name))
    EvalFunc.eval()


    # 差分進化の種類選択
    de = adaptive_DE(
            Z_test=Z_test_tensor,
            EvalFunc=EvalFunc,
            numofpopu=1900,
            Population=Population,
            seed_random=100,
            seed_np=100
        )

    #差分進化の実行
    de.Best_Individual(generation = 0,GENERATION = 1, top_n=BEFORE_TOP)    # 最良個体の選択
    Average_fitness.append(de.Average_Fitness())    # 世代毎に平均評価値を格納
    de.Best_plot(Bestfitness,Bestgene, top_n=BEFORE_TOP)

    Average_fitness_trial.append(Average_fitness[-1])
    Bestgene_trials.append(Bestgene[-BEFORE_TOP:])
    Bestfitness_trials.append(Bestfitness[-BEFORE_TOP:])
    print("************************************************************************************")
    print("************************************************************************************")

    # Bestgeneの各Tensorをnumpy配列に変換
    numpy_arrays = [tensor.numpy() for tensor in Bestgene_trials[0]]

    # numpy配列のリストを結合して1つのnumpy.ndarrayを作成
    Bestgene_conv = np.stack(numpy_arrays)

    # numpy配列のデータ型をnumpy.float32に変換
    Bestgene_conv = Bestgene_conv.astype(np.float32)

    with open(PHASE + "Before_Best_Individual_Latent.pkl",'wb') as f:
        pickle.dump((Bestgene_conv), f)
    with open(PHASE + "Before_Best_Individual_Latent.csv",'w') as f:
        w = csv.writer(f)
        for v in Bestgene_conv:
            w.writerow(v)

    #読み込みからのデコード
    with open(PHASE + "Before_Best_Individual_Latent.pkl","rb") as file:
        before_best = pickle.load(file)

    """差分進化前の最良個体"""
    before_best_tensor = torch.FloatTensor(before_best).cuda()
    # print(before_best_tensor[-1])

    # 上位3体のそれぞれについて処理
    for i, best_individual in enumerate(before_best_tensor):
        # デコード
        best = decoder.decode(best_individual.unsqueeze(0))

        """構造ベクトルの出力"""
        layers = const_model(best[0])
        print(f" {i+1} 位 Architecture:", layers)
        print(f" {i+1} 位 Proxy:", Bestfitness_trials[0][i])

        # DAGのプロット
        plot_DAG(best[0], PHASE, f"before_best{i+1}_p{POPULATION}", data_type=data_type)