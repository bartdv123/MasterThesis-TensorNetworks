using Plots
using LaTeXStrings
using Plots.PlotMeasures

# copy the values because of memory usage 
β_tree = 0.3766

chi_max_list = reverse([(collect(16:8:56))...,
64, 96, 128, 160, 192, 224, 256, 384, 512])


chi_g = [0.4976462313029124,
0.6252136372252755,
0.7662795475736244,
0.7803726882268551,
0.8136730163420332,
0.8678866707766016,
0.8726530514983419,
0.9079002978486277,
0.9359703555289347, 
0.9910619990887413,
0.9901466423510491,
0.9990408663131799,
0.997718713956022,
0.9996110177874771,
0.9997882496843472]

contraction_costs = [826.2834876574206118062948159158013135959186132926265892702914415409645088075244,
512.4422037881374656034707855464277823528230260911482917227388675758732199168455,
367.6667403137365175648261594392110390262518328077610621246510133167292666760401,
195.4868539980316146785305401008295338140478438146504107497941229638258983268729,
105.0876734890634101271416233153232771607046016028280474822744893245224657038981,
 67.73495089079478578745455641030791170385843694137023721051680157470825717556807,
 40.29196376564163335810552955590816879908409825858155743467170144816920079538839,
 16.754253118283889368710707614437503766043344648201337698596019041115150541306,
  7.183240604977202884287062887902464498764737782955389961234860505754514230622454,
  4.924056480607387470624861911744029565950951051479301825777813485448008516279357,
 3.059111816337598168196517163114869343402896338401590776708779400245043886958431,
 2.266856810010645349187538915781228031413822885492196758189888926828288509048552,
 1.597846827485086468355193122702713559764597184003856428384919758169803362323497,
 0.919999196577419808383714624299515937895434551188061140458352581999317090806838,
 0.5415068190491493763432221262578584771124992467913310703597324602807961917769718]



x_values = chi_max_list
# Specify x-ticks where you want the labels
x_ticks = [4, 8, 16, 32, 64, 128, 256, 384, 512]

## Convert tick values to strings for labels
x_tick_labels = string.(x_ticks)

# Create the plot with scatter points and connecting lines
p = plot(x_values, chi_g, seriestype=:scatter, label="Global error", xlabel=L"\chi_{max}", 
    ylabel=L"\mathbf{1} - \textbf{{\frac{|\langle tree | loop \rangle|^2}{\langle loop | loop \rangle \langle tree | tree \rangle}}}", 
    opacity=0.5, left_margin=5mm, xguidefontsize=14, legendfontsize=9, xscale=:log10)

# Set the x-ticks and their labels
xticks!(x_ticks, x_tick_labels)

# Display the plot
display(p)


p = Plots.scatter(chi_max_list, contraction_costs, label="Greedy tree/loop cost", xlabel=L"{\chi_{\text{max}}}", ylabel="Relative contraction costs", color=:red, ylim = [0.1, 1000], legendfontsize=9, xguidefontsize=14, xscale=:log10, yscale=:log10)
Plots.scatter!(chi_max_list, β_tree*contraction_costs, label= "Estimated " * L"\beta_{opt}" * "tree/loop cost", xlabel=L"\chi_{max}", ylabel="Relative contraction cost", color=:orange, opacity = 0.4)
xticks!(x_ticks, x_tick_labels)

display(p)

removed_loop_lengths = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 7, 8, 8, 13]



