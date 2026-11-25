import numpy as np

OUTPUT_CENTERS = np.array([20, 50, 80])

def smoke(x): return np.interp(x,[0,.1,.3,.5,.7,.9,1],[0,.4,.8,.2,.9,.7,.7])
def clean(x): return np.clip(1-np.asarray(x)/.5,0,1)
def fumes(x): return np.clip((np.asarray(x)-.5)/.4,0,1)
def outputs(y):
    y = np.asarray(y)
    return np.array([np.maximum(0, 1 - np.abs(y - center) / 20) for center in OUTPUT_CENTERS])

def controller(x):
    if not 0<=x<=1: raise ValueError('x outside [0,1]')
    strengths=np.array([clean(x),smoke(x),fumes(x)])
    y=np.linspace(0,100,10001)
    z=np.max(np.minimum(strengths[:,None],outputs(y)),axis=0)
    cog=np.trapezoid(y*z,y)/np.trapezoid(z,y)
    tied = np.isclose(strengths, strengths.max(), rtol=0, atol=8 * np.finfo(float).eps)
    maximum = float(OUTPUT_CENTERS[tied][-1])
    return strengths, y, z, cog, maximum

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x = np.linspace(0, 1, 1001)
    figure, ax = plt.subplots(layout="constrained")
    ax.plot(x, clean(x), label="Чисто")
    ax.plot(x, smoke(x), label="Дым")
    ax.plot(x, fumes(x), label="Гарь")
    ax.set(xlabel="Уровень задымлённости x", ylabel="Степень принадлежности")
    ax.legend()
    figure.savefig("input.png", dpi=150)
    plt.close(figure)

    for x in (.1, .9, .3):
        strengths, y, aggregate, centroid, maximum = controller(x)
        print(x, strengths, centroid, maximum)
        figure, ax = plt.subplots(layout="constrained")
        ax.plot(y, aggregate, label="Агрегация g(y)")
        ax.axvline(centroid, color="tab:orange", linestyle="--", label="Центр тяжести")
        ax.axvline(maximum, color="tab:green", linestyle=":", label="Метод максимума")
        ax.set(xlabel="Интенсивность вентиляции, %", ylabel="Степень принадлежности")
        ax.legend()
        figure.savefig(f"output_{x}.png", dpi=150)
        plt.close(figure)
