// assets/charts.js — MTL vs Single-Task Performance Comparison
(function() {
  var style = getComputedStyle(document.documentElement);
  var accent = style.getPropertyValue('--accent').trim();
  var accent2 = style.getPropertyValue('--accent2').trim();
  var ink = style.getPropertyValue('--ink').trim();
  var muted = style.getPropertyValue('--muted').trim();
  var rule = style.getPropertyValue('--rule').trim();
  var gainPos = style.getPropertyValue('--gain-pos').trim();
  var gainNeg = style.getPropertyValue('--gain-neg').trim();

  var tasks = [
    'is_click', 'is_like', 'is_follow', 'is_comment',
    'is_forward', 'long_view', 'is_hate', 'is_profile_enter'
  ];

  var labels = [
    '点击\n(34.8%)', '点赞\n(1.4%)', '关注\n(0.1%)', '评论\n(0.2%)',
    '转发\n(0.09%)', '长时观看\n(23.7%)', '负反馈\n(0.03%)', '进入主页\n(1.6%)'
  ];

  // Single-Task AUC
  var st   = [0.7144, 0.7523, 0.6596, 0.7143, 0.6307, 0.7327, 0.6394, 0.6375];
  // MTL AUCs
  var sb   = [0.7044, 0.7726, 0.7484, 0.6572, 0.5879, 0.7244, 0.5834, 0.6543];
  var mmoe = [0.6810, 0.7375, 0.7304, 0.7129, 0.7109, 0.7042, 0.5611, 0.6485];
  var ple  = [0.6958, 0.7756, 0.7503, 0.7778, 0.5925, 0.7169, 0.5898, 0.6594];

  // MTL Gain = MTL AUC - Single-Task AUC
  var gain_sb   = sb.map(function(v, i) { return v - st[i]; });
  var gain_mmoe = mmoe.map(function(v, i) { return v - st[i]; });
  var gain_ple  = ple.map(function(v, i) { return v - st[i]; });

  // ============================================================
  // Chart 1: Per-task AUC — 4 models
  // ============================================================
  var chart1 = echarts.init(document.getElementById('chart-auc'), null, { renderer: 'svg' });
  chart1.setOption({
    animation: false,
    tooltip: {
      trigger: 'axis',
      appendToBody: true,
      formatter: function(params) {
        var s = params[0].axisValue.replace('\n', ' ') + '<br/>';
        params.forEach(function(p) {
          s += '<span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:' + p.color + ';margin-right:6px;"></span>';
          s += p.seriesName + ': <b>' + p.value.toFixed(4) + '</b><br/>';
        });
        return s;
      }
    },
    legend: {
      data: ['Single-Task', 'SharedBottom (MTL)', 'MMoE (MTL)', 'PLE (MTL)'],
      bottom: 0,
      textStyle: { color: ink, fontSize: 12 }
    },
    grid: {
      left: '3%', right: '4%', top: '8%', bottom: '14%',
      containLabel: true
    },
    xAxis: {
      type: 'category',
      data: labels,
      axisLabel: {
        color: ink,
        fontSize: 11,
        interval: 0
      },
      axisLine: { lineStyle: { color: rule } },
      axisTick: { show: false }
    },
    yAxis: {
      type: 'value',
      name: 'AUC',
      min: 0.50,
      max: 0.82,
      interval: 0.05,
      axisLabel: {
        color: muted,
        fontSize: 11,
        formatter: function(v) { return v.toFixed(2); }
      },
      splitLine: { lineStyle: { color: rule, type: 'dashed' } },
      nameTextStyle: { color: muted, fontSize: 11 }
    },
    series: [
      {
        name: 'Single-Task',
        type: 'bar',
        data: st,
        itemStyle: { color: '#eab308', borderRadius: [3, 3, 0, 0] },
        barGap: '10%',
        barWidth: '18%'
      },
      {
        name: 'SharedBottom (MTL)',
        type: 'bar',
        data: sb,
        itemStyle: { color: muted, borderRadius: [3, 3, 0, 0] }
      },
      {
        name: 'MMoE (MTL)',
        type: 'bar',
        data: mmoe,
        itemStyle: { color: accent2, borderRadius: [3, 3, 0, 0] }
      },
      {
        name: 'PLE (MTL)',
        type: 'bar',
        data: ple,
        itemStyle: { color: accent, borderRadius: [3, 3, 0, 0] }
      }
    ]
  });
  window.addEventListener('resize', function() { chart1.resize(); });

  // ============================================================
  // Chart 2: MTL Gain — 3 models grouped bar
  // ============================================================
  var chart2 = echarts.init(document.getElementById('chart-gain'), null, { renderer: 'svg' });
  chart2.setOption({
    animation: false,
    tooltip: {
      trigger: 'axis',
      appendToBody: true,
      formatter: function(params) {
        var s = params[0].axisValue.replace('\n', ' ') + '<br/>';
        params.forEach(function(p) {
          var sign = p.value >= 0 ? '+' : '';
          s += '<span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:' + p.color + ';margin-right:6px;"></span>';
          s += p.seriesName + ': <b>' + sign + p.value.toFixed(4) + '</b><br/>';
        });
        return s;
      }
    },
    legend: {
      data: ['SharedBottom (MTL)', 'MMoE (MTL)', 'PLE (MTL)'],
      bottom: 0,
      textStyle: { color: ink, fontSize: 12 }
    },
    grid: {
      left: '3%', right: '4%', top: '8%', bottom: '14%',
      containLabel: true
    },
    xAxis: {
      type: 'category',
      data: labels,
      axisLabel: {
        color: ink,
        fontSize: 11,
        interval: 0
      },
      axisLine: { lineStyle: { color: rule } },
      axisTick: { show: false }
    },
    yAxis: {
      type: 'value',
      name: 'MTL Gain',
      axisLabel: {
        color: muted,
        fontSize: 11,
        formatter: function(v) { return (v >= 0 ? '+' : '') + v.toFixed(3); }
      },
      splitLine: { lineStyle: { color: rule, type: 'dashed' } },
      nameTextStyle: { color: muted, fontSize: 11 }
    },
    series: [
      {
        name: 'SharedBottom (MTL)',
        type: 'bar',
        data: gain_sb.map(function(v) {
          return {
            value: v,
            itemStyle: {
              color: v >= 0 ? gainPos : gainNeg,
              borderRadius: v >= 0 ? [3, 3, 0, 0] : [3, 3, 0, 0]
            }
          };
        }),
        barGap: '10%',
        barWidth: '22%'
      },
      {
        name: 'MMoE (MTL)',
        type: 'bar',
        data: gain_mmoe.map(function(v) {
          return {
            value: v,
            itemStyle: {
              color: v >= 0 ? gainPos : gainNeg
            }
          };
        })
      },
      {
        name: 'PLE (MTL)',
        type: 'bar',
        data: gain_ple.map(function(v) {
          return {
            value: v,
            itemStyle: {
              color: v >= 0 ? gainPos : gainNeg
            }
          };
        })
      }
    ]
  });
  window.addEventListener('resize', function() { chart2.resize(); });
})();