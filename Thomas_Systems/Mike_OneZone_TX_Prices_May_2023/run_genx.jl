using DataFrames
using CSV

#time_index column
time_index = []

for i in 0:24*(31 + 2)
    push!(time_index, i)
end


for i in 0:0
    fuel_year = CSV.File("/Users/Michael/GenX-main/Example_Systems/OneZone_TX_Prices_Delinearized/Yearly_Data_Files/Fuels_data.csv") |> DataFrame
    load_year = CSV.File("/Users/Michael/GenX-main/Example_Systems/OneZone_TX_Prices_Delinearized/Yearly_Data_Files/Load_data.csv") |> DataFrame
    var_year = CSV.File("/Users/Michael/GenX-main/Example_Systems/OneZone_TX_Prices_Delinearized/Yearly_Data_Files/Generators_variability.csv") |> DataFrame

    println(names(fuel_year))
    fuel_excerpt = fuel_year[1+ i * 7 * 24 : (i + 1) * 31 * 24 + 1 + 48, :]
    fuel_excerpt[1, :] = fuel_year[1, :]
    fuel_excerpt[:, :Time_Index] = time_index
    fuel_excerpt[:, :Column1] = time_index[1: end ]
    fuel_excerpt[:, "Unnamed: 0"] = time_index[1: end ]

    load_excerpt = load_year[1+ i * 7 * 24 : (i + 1) * 31 * 24 + 48, :]
    load_excerpt[:, :Time_Index] = time_index[2: end]
    #load_excerpt[:, :Column1] = time_index[1: end - 1]
    #load_excerpt[:, "Unnamed: 0"] = time_index[1: end - 1]

    var_excerpt = var_year[1+ i * 7 * 24 : (i + 1) * 31 * 24 + 48, :]
    var_excerpt[:, :Time_Index] = time_index[2: end]
    var_excerpt[:, :Column1] = time_index[1: end - 1]
    #var_excerpt[:, "Unnamed: 0"] = time_index[1: end - 1]

    CSV.write("/Users/Michael/GenX-main/Example_Systems/OneZone_TX_Prices_Delinearized/Fuels_data.csv", fuel_excerpt)
    CSV.write("/Users/Michael/GenX-main/Example_Systems/OneZone_TX_Prices_Delinearized/Load_data.csv", load_excerpt)
    CSV.write("/Users/Michael/GenX-main/Example_Systems/OneZone_TX_Prices_Delinearized/Generators_variability.csv", var_excerpt)

    cd("/Users/Michael/GenX-main")
    cmd = `julia --project=. "/Users/Michael/GenX-main/Example_Systems/OneZone_TX_Prices_Delinearized/Run.jl"`
    run(cmd)
    
end

