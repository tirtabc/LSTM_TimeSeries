var ctx = document.getElementById('ChartKelembaban').getContext('2d');
var config = {
  "type": "line",
  "data": {
    "labels": [],
    "datasets": [
      {
        "label": "Kelembaban",
        "data": [],
        "backgroundColor": "rgba(1, 22, 39, 1)",
        "borderColor": "rgba(1, 22, 39, 1)",
        "fill": false,
        "pointBorderWidth": [],
        "cubicInterpolationMode": "monotone"
      },{
        "label": "P-Kelembaban",
        "data": [],
        "backgroundColor": "rgba(1, 22, 39, 0.5)",
        "borderColor": "rgba(1, 22, 39, 0.5)",
        "fill": false,
        "pointBorderWidth": [],
        "cubicInterpolationMode": "monotone"
      },{
        "label": "Batas Bawah",
        "data": [],
        "backgroundColor": "rgba(235, 64, 52, 0.2)",
        "borderColor": "rgba(235, 64, 52, 0.2)",
        "fill": false,
        "pointBorderWidth": [],
        "cubicInterpolationMode": "monotone"
      }, {
        "label": "Batas Atas",
        "data": [],
        "backgroundColor": "rgba(4, 101, 212, 0.4)",
        "borderColor": "rgba(4, 101, 212, 0.4)",
        "fill": false,
        "pointBorderWidth": [],
        "cubicInterpolationMode": "monotone"
      }
    ]
  },
  "options": {
    "scales": {
        yAxes: [{
            display: true,
            ticks: {
                min: 60, // minimum value
//                min: Math.min.apply(this, config.data.datasets[1]) - 5,
                max: 80 // maximum value
            }
        }]
    }
  }
}

var ctx1 = document.getElementById('ChartSuhuTanah').getContext('2d');
var config1 = {
  "type": "line",
  "data": {
    "labels": [],
    "datasets": [
      {
        "label": "suhu_tanah",
        "data": [],
        "backgroundColor": "rgba(1, 22, 39, 1)",
        "borderColor": "rgba(1, 22, 39, 1)",
        "fill": false,
        "pointBorderWidth": [],
        "cubicInterpolationMode": "monotone"
      }
    ]
  },
  "options": {
    "scales": {
        yAxes: [{
            display: true,
            ticks: {
                min: 10, // minimum value
                max: 30 // maximum value
            }
        }]
    }
  }
}

var ctx2 = document.getElementById('ChartSuhuPermukaan').getContext('2d');
var config2 = {
  "type": "line",
  "data": {
    "labels": [],
    "datasets": [
      {
        "label": "suhu_permukaan",
        "data": [],
        "backgroundColor": "rgba(1, 22, 39, 1)",
        "borderColor": "rgba(1, 22, 39, 1)",
        "fill": false,
        "pointBorderWidth": [],
        "cubicInterpolationMode": "monotone"
      }
    ]
  },
  "options": {
    "scales": {
        yAxes: [{
            display: true,
            ticks: {
                min: 110, // minimum value
                max: 150 // maximum value
            }
        }],
        xAxes: {
            type: 'time',
            time: {
                displayFormats: {
                    quarter: 'HH : MM : II'
                             }
            }
        }
  }
}
}

var chart = new Chart(ctx, config);
var chart1 = new Chart(ctx1, config1);
var chart2 = new Chart(ctx2, config2);
