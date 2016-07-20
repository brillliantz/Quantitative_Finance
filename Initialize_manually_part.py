from sys import path
path.append('/home/bingnan/ecworkspace/HFT1')
import Initialize_module_part
# %matplotlib inline

# Style. 1
sns.set_context('paper')
sns.set_style("darkgrid")
# Style. 2
sns.set_context('paper')
sns.set_style("dark", 
    rc={'axes.facecolor': 'black', 
    'grid.color': 'red', 
    'grid.linestyle': '--', 
    'figure.facecolor': 'grey'})

hft = pd.read_hdf('/home/bingnan/HFT_SR_RM_MA_TA.hdf')
ta = hft.minor_xs('TA0001')

ta = AddCol(ta)
ta_pm = GiveMePM(ta, nforward=60, nbackward=100, lim=[0, 30], cutdepth=0, norm=False, high_var_length=200)
selected_arr = CuthlLimit(df, forward=60, backward=100, how='all', depth=0)



insample_index0, outsample_index0 = GiveMeIndex([[0,2], [4, 6], [8, 10], [14, 16], [18,20], [22, 24], [26,28]],
                                               [[2, 4], [6, 8], [12, 14], [16, 18], [20,22], [24, 26], [28,30]])
insample_index1, outsample_index1 = GiveMeIndex([[0, 12], [14,20]], [[20, 25]])
insample_index2, outsample_index2 = GiveMeIndex([[5, 25]], [[0, 5]])
insample_index3, outsample_index3 = GiveMeIndex([[0, 5], [10, 25]], [[5, 10]])
insample_index4, outsample_index4 = GiveMeIndex([[0, 10], [15, 25]], [[10, 15]])
insample_index5, outsample_index5 = GiveMeIndex([[0, 13]], [[13, 25]])
insample_index6, outsample_index6 = GiveMeIndex([[0, 18]], [[18, 25]])
# different_io_index = ((insample_index0, outsample_index0), (insample_index1, outsample_index1), 
#                       (insample_index2, outsample_index2), (insample_index3, outsample_index3),
#                       (insample_index4, outsample_index4), (insample_index5, outsample_index5))
insample_index00, outsample_index00 = GiveMeIndex([[0, 20]],
                                               [[25, 30]])

plt.plot(ta_pm)