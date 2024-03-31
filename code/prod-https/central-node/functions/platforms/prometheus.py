from prometheus_client import Gauge

# Created and works
def central_global_gauge(
    prometheus_registry: any
) -> any:
    gauge = Gauge(
        name = 'C_M_M',
        documentation = 'Central global metrics',
        labelnames = [
            'date',
            'time',
            'collector',
            'name', 
            'experiment',
            'cycle',
            'source',
            'metric'
        ],
        registry = prometheus_registry
    )
    names = {
        'train-amount': 'TrAm',
        'test-amount': 'TeAm',
        'eval-amount': 'EvAm',
        'true-positives': 'TrPo',
        'false-positives': 'FaPo',
        'true-negatives': 'TrNe',
        'false-negatives': 'FaNe',
        'recall': 'ReMe',
        'selectivity': 'SeMe',
        'precision': 'PrMe',
        'miss-rate': 'MiRaMe',
        'fall-out': 'FaOuMe',
        'balanced-accuracy': 'BaAcMe',
        'accuracy': 'AcMe'
    }
    return gauge, names
# Created and works
def central_resource_gauge(
    prometheus_registry: any 
) -> any:
    gauge = Gauge(
        name = 'C_R_M',
        documentation = 'Central resource metrics',
        labelnames = [
            'date',
            'time',
            'collector',
            'name', 
            'experiment',
            'cycle',
            'source',
            'metric'
        ],
        registry = prometheus_registry
    )
    names = {
        'physical-cpu-amount': 'PyCPUAm',
        'total-cpu-amount': 'ToCPUAm',
        'min-cpu-frequency-mhz': 'MinCPUFrMhz',
        'max-cpu-frequency-mhz': 'MaxCPUFrMhz',
        'total-ram-amount-bytes': 'ToRAMAmBy',
        'available-ram-amount-bytes': 'AvRAMAmByte',
        'total-disk-amount-bytes': 'ToDiAmBy',
        'available-disk-amount-bytes': 'ToDiAmByte',
        'cpu-percent': 'CPUPer',
        'ram-percent': 'RAMPer',
        'ram-total-bytes': 'RAMToBy',
        'ram-free-bytes': 'RAMFrBy',
        'ram-used-bytes': 'RAMUsBy',
        'disk-total-bytes': 'DiToBy',
        'disk-free-bytes': 'DiFrBy',
        'disk-used-bytes': 'DiUsBy',
        'network-sent-bytes': 'NeSeBy',
        'network-received-bytes': 'NeReBy',
        'network-packets-sent': 'NePaSe',
        'network-packets-received': 'NePaRe',
        'network-packets-sending-errors': 'NePaSeEr',
        'network-packets-reciving-errors': 'NePaReEr',
        'network-packets-outgoing-dropped': 'NePaOuDr',
        'network-packets-incoming-dropped': 'NePaInDr',
        'ram-resident-set-size-bytes': 'RAMReSeSiBy',
        'ram-virtual-memory-size-bytes': 'RAMViMeSiBy',
        'ram-shared-memory-bytes': 'RAMShMeBy',
        'ram-code-segment-size-bytes': 'RAMCoSeSiBy',
        'ram-data-segment-size-bytes': 'RAMDaSeSiBy',
        'ram-library-size-bytes': 'RAMLiSiBy',
        'ram-dirty-pages-bytes': 'RAMDiPaBy',
        'ram-unique-set-size-bytes': 'RAMUnSeSiBy',
        'disk-read-bytes': 'DiReBy',
        'disk-write-bytes': 'DiWrBy',
        'disk-read-characters-bytes': 'DiReChBy',
        'disk-write-characters-bytes': 'DiWrChBy'
    }
    return gauge, names
# Created and works
def central_time_gauge(
    prometheus_registry: any    
) -> any:
    gauge = Gauge(
        name = 'C_T_M',
        documentation = 'Central time metrics',
        labelnames = [
            'date', 
            'time',
            'collector', 
            'name', 
            'experiment',
            'cycle',
            'area',
            'source',
            'metric'
        ],
        registry = prometheus_registry
    )
    names = {
        'experiment-date': 'ExDa',
        'experiment-time-start':'ExTiSt',
        'experiment-time-end':'ExTiEn',
        'experiment-total-seconds': 'ExToSec',
        'cycle-time-start': 'CyTiSt',
        'cycle-time-end': 'CyTiEn',
        'cycle-total-seconds': 'CyToSec',
        'action-time-start': 'AcTiSt',
        'action-time-end': 'AcTiEn',
        'action-total-seconds': 'AcToSec',
        'processing-time-seconds': 'PrTiSec',
        'elapsed-time-seconds': 'ElTiSec',
        'epochs': 'Epo',
        'batches': 'Bat',
        'average-batch-size': 'AvBatSi'
    }
    return gauge, names