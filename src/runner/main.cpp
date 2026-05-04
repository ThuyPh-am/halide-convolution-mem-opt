#include "../config/layer_definitions.h"
#include "../generator/schedule_baseline.h"
#include "../generator/schedule_os_inspired.h"
#include "../generator/schedule_rs_inspired.h"
#include "../generator/schedule_ws_inspired.h"

int main(){
    for (const auto& model : ALL_MODELS){
        for (const auto& layer : model.layers){
            measure_layer_baseline(layer, model.name);
            measure_layer_rs(layer,model.name);
            measure_layer_ws(layer,model.name);
            measure_layer_os(layer, model.name);
        }
    }
    return 0;
}