import React, { useState, useEffect, useCallback } from 'react';
import { Upload, Play, Download, TrendingUp, TrendingDown, Minus, Filter, Search, RefreshCw, AlertCircle, CheckCircle, BarChart3, Settings, Database, Wifi, WifiOff } from 'lucide-react';

const StockAnalysisDashboard = () => {
  const [screenedStocks, setScreenedStocks] = useState([]);
  const [analysisResults, setAnalysisResults] = useState([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisProgress, setAnalysisProgress] = useState({ current: 0, total: 0 });
  const [analysisMode, setAnalysisMode] = useState('top20');
  const [customLimit, setCustomLimit] = useState(10);
  const [filterAction, setFilterAction] = useState('all');
  const [searchTerm, setSearchTerm] = useState('');
  const [sortBy, setSortBy] = useState('fitness_score');
  const [sortOrder, setSortOrder] = useState('desc');
  const [systemStatus, setSystemStatus] = useState(null);
  const [serverConnected, setServerConnected] = useState(false);
  const [uploadingFile, setUploadingFile] = useState(false);

  // API基底URL
  const API_BASE = 'http://localhost:5000/api';

  // システム状態確認
  const checkSystemStatus = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/status`);
      const data = await response.json();

      if (data.status === 'success') {
        setSystemStatus(data);
        setServerConnected(true);
        return true;
      } else {
        setServerConnected(false);
        return false;
      }
    } catch (error) {
      console.error('システム状態確認エラー:', error);
      setServerConnected(false);
      return false;
    }
  }, []);

  // スクリーニング結果取得
  const loadScreenedStocks = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/stocks`);
      const data = await response.json();

      if (data.status === 'success') {
        setScreenedStocks(data.data);
        return true;
      } else {
        console.error('スクリーニング結果取得エラー:', data.message);
        return false;
      }
    } catch (error) {
      console.error('スクリーニング結果取得エラー:', error);
      return false;
    }
  }, []);

  // 初期化
  useEffect(() => {
    const initialize = async () => {
      const connected = await checkSystemStatus();
      if (connected) {
        await loadScreenedStocks();
      }
    };
    initialize();
  }, [checkSystemStatus, loadScreenedStocks]);

  // CSVファイルアップロード
  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setUploadingFile(true);
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(`${API_BASE}/upload_screener`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (data.status === 'success') {
        alert(`✅ ${data.message}`);
        await loadScreenedStocks();
        setAnalysisResults([]);
      } else {
        alert(`❌ アップロードエラー: ${data.message}`);
      }
    } catch (error) {
      console.error('ファイルアップロードエラー:', error);
      alert('❌ ファイルアップロードに失敗しました');
    } finally {
      setUploadingFile(false);
      event.target.value = '';
    }
  };

  // サンプルデータ使用
  const useSampleData = () => {
    const sampleData = [
      { ティッカー: '5721', 銘柄名: 'エス・サイエンス', 適性スコア: 60.4 },
      { ティッカー: '5715', 銘柄名: '古河機械金属', 適性スコア: 60.4 },
      { ティッカー: '5724', 銘柄名: 'アサカ理研', 適性スコア: 60.4 },
      { ティッカー: '5803', 銘柄名: 'フジクラ', 適性スコア: 44.1 },
      { ティッカー: '7603', 銘柄名: 'マックハウス', 適性スコア: 41.2 },
      { ティッカー: '3350', 銘柄名: 'メタプラネット', 適性スコア: 31.5 },
      { ティッカー: '6836', 銘柄名: 'ぷらっとホーム', 適性スコア: 25.7 },
      { ティッカー: '7203', 銘柄名: 'トヨタ自動車', 適性スコア: 24.5 },
      { ティッカー: '7208', 銘柄名: 'カネミツ', 適性スコア: 24.5 },
      { ティッカー: '4584', 銘柄名: 'キッズウェル・バイオ', 適性スコア: 23.7 },
      { ティッカー: '3905', 銘柄名: 'データセクション', 適性スコア: 22.6 },
      { ティッカー: '6232', 銘柄名: 'ＡＣＳＬ', 適性スコア: 21.5 },
    ];
    setScreenedStocks(sampleData);
    setAnalysisResults([]);
  };

  // 分析実行
  const runAnalysis = async () => {
    if (screenedStocks.length === 0 || isAnalyzing || !serverConnected) {
      return;
    }

    setIsAnalyzing(true);
    setAnalysisResults([]);

    let targetStocks = [...screenedStocks];
    switch (analysisMode) {
      case 'top10':
        targetStocks = targetStocks.slice(0, 10);
        break;
      case 'top20':
        targetStocks = targetStocks.slice(0, 20);
        break;
      case 'custom':
        targetStocks = targetStocks.slice(0, customLimit);
        break;
      default:
        break;
    }

    setAnalysisProgress({ current: 0, total: targetStocks.length });

    try {
      // バッチ分析を実行
      const response = await fetch(`${API_BASE}/analyze_batch`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          stocks: targetStocks
        }),
      });

      const data = await response.json();

      if (data.status === 'success') {
        const results = data.data.results.map(result => ({
          ticker: result.ticker,
          name: result.name,
          fitness_score: result.fitness_score,
          action: result.action,
          price: Math.round(result.current_price || 0),
          change_pct: result.price_change || 0,
          confidence: result.confidence || 0,
          volatility: result.volatility || 0,
          error: result.error
        }));

        setAnalysisResults(results);
        setAnalysisProgress({ current: targetStocks.length, total: targetStocks.length });
      } else {
        alert(`❌ 分析エラー: ${data.message}`);
      }
    } catch (error) {
      console.error('分析実行エラー:', error);
      alert('❌ 分析の実行に失敗しました。APIサーバーが起動していることを確認してください。');
    } finally {
      setIsAnalyzing(false);
    }
  };

  // フィルタリングとソート
  const getFilteredResults = () => {
    return analysisResults
      .filter(result => {
        if (filterAction !== 'all' && result.action !== filterAction) return false;
        if (searchTerm &&
            !result.name.toLowerCase().includes(searchTerm.toLowerCase()) &&
            !result.ticker.includes(searchTerm)) return false;
        return true;
      })
      .sort((a, b) => {
        const multiplier = sortOrder === 'desc' ? -1 : 1;
        const valA = a[sortBy] || 0;
        const valB = b[sortBy] || 0;
        return (valA - valB) * multiplier;
      });
  };

  // UI用の補助関数
  const getMarketStatus = () => {
    if (!systemStatus) return { status: '⚪ 不明', color: 'text-gray-600 bg-gray-50' };

    const status = systemStatus.market.status;
    if (status.includes('開場中')) return { status, color: 'text-green-600 bg-green-50' };
    if (status.includes('開場前')) return { status, color: 'text-yellow-600 bg-yellow-50' };
    return { status, color: 'text-red-600 bg-red-50' };
  };

  const getActionStyle = (action) => {
    switch (action) {
      case 'BUY': return { color: 'text-green-700 bg-green-100 border-green-200', icon: <TrendingUp className="w-4 h-4" /> };
      case 'SELL': return { color: 'text-red-700 bg-red-100 border-red-200', icon: <TrendingDown className="w-4 h-4" /> };
      case 'HOLD': return { color: 'text-gray-700 bg-gray-100 border-gray-200', icon: <Minus className="w-4 h-4" /> };
      default: return { color: 'text-orange-700 bg-orange-100 border-orange-200', icon: <AlertCircle className="w-4 h-4" /> };
    }
  };

  const getAnalysisSummary = () => {
    const buyCount = analysisResults.filter(r => r.action === 'BUY').length;
    const sellCount = analysisResults.filter(r => r.action === 'SELL').length;
    const holdCount = analysisResults.filter(r => r.action === 'HOLD').length;
    const errorCount = analysisResults.filter(r => r.action === 'ERROR').length;
    const avgConfidence = analysisResults.length > 0 ?
      analysisResults.filter(r => r.action !== 'ERROR').reduce((sum, r) => sum + r.confidence, 0) / Math.max(1, analysisResults.length - errorCount) : 0;
    return { buyCount, sellCount, holdCount, errorCount, avgConfidence };
  };

  const exportResults = () => {
    const filtered = getFilteredResults();
    const csvContent = [
      ...filtered.map(r => [
        r.ticker.replace(/\.T$/, ''), `"${r.name}"`
      ].join(','))
    ].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `analysis_${new Date().toISOString().split('T')[0]}.csv`;
    link.click();
  };

  const marketStatus = getMarketStatus();
  const summary = getAnalysisSummary();
  const filteredResults = getFilteredResults();

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 p-6">
      <div className="max-w-7xl mx-auto">
        <div className="bg-white rounded-2xl shadow-2xl border border-slate-200 overflow-hidden">
          <div className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white p-8 text-center">
            <h1 className="text-4xl font-bold mb-2">🚀 スクリーニング銘柄DQN分析システム</h1>
            <p className="text-xl text-blue-100 mb-4">
              📈 N225最適モデル (Sharpe Ratio: {systemStatus?.system.performance || '1.5139'})
            </p>
            <div className="flex items-center justify-center gap-4">
              <div className={`inline-flex items-center px-4 py-2 rounded-full text-sm font-semibold ${marketStatus.color}`}>
                {marketStatus.status}
              </div>
              <div className={`inline-flex items-center px-4 py-2 rounded-full text-sm font-semibold ${
                serverConnected ? 'text-green-700 bg-green-100' : 'text-red-700 bg-red-100'
              }`}>
                {serverConnected ? <Wifi className="w-4 h-4 mr-2" /> : <WifiOff className="w-4 h-4 mr-2" />}
                {serverConnected ? 'API接続済み' : 'API切断中'}
              </div>
            </div>
          </div>

          <div className="p-8">
            {!serverConnected && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
                <div className="flex items-center">
                  <AlertCircle className="w-5 h-5 text-red-600 mr-3" />
                  <div>
                    <p className="text-red-800 font-semibold">APIサーバーに接続できません</p>
                    <p className="text-red-600 text-sm mt-1">
                      Python APIサーバー (http://localhost:5000) が起動していることを確認してください。<br />
                      <code className="bg-red-100 px-2 py-1 rounded">python simple.py</code> でサーバーを起動してください。
                    </p>
                  </div>
                </div>
                <button
                  onClick={checkSystemStatus}
                  className="mt-3 bg-red-600 text-white px-4 py-2 rounded-lg hover:bg-red-700 transition-colors text-sm"
                >
                  再接続試行
                </button>
              </div>
            )}

            <div className="grid lg:grid-cols-3 gap-8">
              {/* Data Management */}
              <div className="bg-slate-50 rounded-xl p-6 border border-slate-200">
                <h3 className="text-lg font-semibold mb-4 flex items-center text-slate-800">
                  <Database className="mr-2 w-5 h-5" />データ管理
                </h3>
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-2">CSVファイルアップロード</label>
                    <input
                      type="file"
                      accept=".csv"
                      onChange={handleFileUpload}
                      disabled={!serverConnected || uploadingFile}
                      className="w-full p-3 border border-slate-300 rounded-lg bg-white focus:ring-2 focus:ring-blue-500 disabled:bg-slate-100 disabled:cursor-not-allowed"
                    />
                    {uploadingFile && <p className="text-sm text-blue-600 mt-1">📤 アップロード中...</p>}
                  </div>
                  <button
                    onClick={useSampleData}
                    disabled={!serverConnected}
                    className="w-full bg-slate-600 text-white py-3 px-4 rounded-lg hover:bg-slate-700 disabled:bg-slate-400 disabled:cursor-not-allowed transition-colors font-medium"
                  >
                    サンプルデータを使用
                  </button>
                  {screenedStocks.length > 0 && (
                    <div className="bg-green-50 border border-green-200 rounded-lg p-3 text-green-800">
                      <p className="font-medium">✅ {screenedStocks.length}銘柄読み込み済み</p>
                    </div>
                  )}
                </div>
              </div>

              {/* Analysis Settings */}
              <div className="bg-slate-50 rounded-xl p-6 border border-slate-200">
                <h3 className="text-lg font-semibold mb-4 flex items-center text-slate-800">
                  <Settings className="mr-2 w-5 h-5" />分析設定
                </h3>
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-2">分析モード</label>
                    <select
                      value={analysisMode}
                      onChange={(e) => setAnalysisMode(e.target.value)}
                      className="w-full p-3 border border-slate-300 rounded-lg bg-white focus:ring-2 focus:ring-blue-500"
                    >
                      <option value="top10">⚡ クイック分析（上位10銘柄）</option>
                      <option value="top20">📊 上位分析（上位20銘柄）</option>
                      <option value="custom">🎯 カスタム分析</option>
                      <option value="all">🔄 完全分析（全銘柄）</option>
                    </select>
                  </div>
                  {analysisMode === 'custom' && (
                    <div>
                      <label className="block text-sm font-medium text-slate-700 mb-2">分析銘柄数</label>
                      <input
                        type="number"
                        value={customLimit}
                        onChange={(e) => setCustomLimit(Math.max(1, parseInt(e.target.value) || 10))}
                        min="1"
                        max="100"
                        className="w-full p-3 border border-slate-300 rounded-lg bg-white focus:ring-2 focus:ring-blue-500"
                      />
                    </div>
                  )}
                </div>
              </div>

              {/* System Status */}
              <div className="bg-slate-50 rounded-xl p-6 border border-slate-200">
                <h3 className="text-lg font-semibold mb-4 flex items-center text-slate-800">
                  <BarChart3 className="mr-2 w-5 h-5" />システム状況
                </h3>
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-slate-600">DQNモデル</span>
                    <span className={`font-semibold ${systemStatus?.system.model_loaded ? 'text-green-600' : 'text-red-600'}`}>
                      {systemStatus?.system.model_loaded ? '✅ 読み込み済み' : '❌ 未読み込み'}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-slate-600">特徴量数</span>
                    <span className="font-semibold">{systemStatus?.system.feature_count || 29}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-slate-600">スクリーニングデータ</span>
                    <span className="font-semibold">{systemStatus?.data.stock_count || 0}銘柄</span>
                  </div>
                </div>
              </div>
            </div>

            <div className="mt-8 text-center">
              <button
                onClick={runAnalysis}
                disabled={screenedStocks.length === 0 || isAnalyzing || !serverConnected}
                className="bg-gradient-to-r from-green-500 to-emerald-500 text-white py-4 px-8 rounded-xl hover:from-green-600 hover:to-emerald-600 disabled:from-slate-400 disabled:to-slate-500 disabled:cursor-not-allowed transition-all duration-300 font-semibold text-lg shadow-lg flex items-center mx-auto"
              >
                {isAnalyzing ? (
                  <><RefreshCw className="mr-3 w-6 h-6 animate-spin" />分析中... ({analysisProgress.current}/{analysisProgress.total})</>
                ) : (
                  <><Play className="mr-3 w-6 h-6" />DQN分析開始</>
                )}
              </button>
            </div>
          </div>
        </div>

        {analysisResults.length > 0 && (
          <div className="mt-8 bg-white rounded-2xl shadow-2xl border border-slate-200 overflow-hidden">
            <div className="bg-gradient-to-r from-slate-700 to-slate-800 text-white p-6 flex flex-col lg:flex-row lg:items-center lg:justify-between">
              <div>
                <h2 className="text-2xl font-bold mb-2">📊 DQN分析結果</h2>
                <p className="text-slate-300">{new Date().toLocaleDateString()} • {analysisResults.length}銘柄分析完了</p>
              </div>
              <div className="flex flex-wrap gap-3 mt-4 lg:mt-0">
                <div className="bg-green-600 bg-opacity-80 px-4 py-2 rounded-lg">
                  <span className="font-semibold">🟢 買い: {summary.buyCount}</span>
                </div>
                <div className="bg-red-600 bg-opacity-80 px-4 py-2 rounded-lg">
                  <span className="font-semibold">🔴 売り: {summary.sellCount}</span>
                </div>
                <div className="bg-slate-600 bg-opacity-80 px-4 py-2 rounded-lg">
                  <span className="font-semibold">⚪ 様子見: {summary.holdCount}</span>
                </div>
                {summary.errorCount > 0 && (
                  <div className="bg-orange-600 bg-opacity-80 px-4 py-2 rounded-lg">
                    <span className="font-semibold">❌ エラー: {summary.errorCount}</span>
                  </div>
                )}
                <div className="bg-blue-600 bg-opacity-80 px-4 py-2 rounded-lg">
                  <span className="font-semibold">平均信頼度: {(summary.avgConfidence * 100).toFixed(0)}%</span>
                </div>
              </div>
            </div>

            <div className="p-6">
              <div className="grid md:grid-cols-5 gap-4 mb-6">
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-1">アクション</label>
                  <select value={filterAction} onChange={(e) => setFilterAction(e.target.value)} className="w-full p-3 border border-slate-300 rounded-lg bg-white focus:ring-2 focus:ring-blue-500">
                    <option value="all">すべて</option>
                    <option value="BUY">買い</option>
                    <option value="SELL">売り</option>
                    <option value="HOLD">様子見</option>
                    <option value="ERROR">エラー</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-1">検索</label>
                  <div className="relative">
                    <input type="text" value={searchTerm} onChange={(e) => setSearchTerm(e.target.value)} placeholder="銘柄名・ティッカー" className="w-full p-3 pl-10 border border-slate-300 rounded-lg bg-white focus:ring-2 focus:ring-blue-500" />
                    <Search className="absolute left-3 top-3.5 w-4 h-4 text-slate-400" />
                  </div>
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-1">ソート</label>
                  <select value={sortBy} onChange={(e) => setSortBy(e.target.value)} className="w-full p-3 border border-slate-300 rounded-lg bg-white focus:ring-2 focus:ring-blue-500">
                    <option value="fitness_score">適性スコア</option>
                    <option value="confidence">信頼度</option>
                    <option value="price">価格</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-1">並び順</label>
                  <select value={sortOrder} onChange={(e) => setSortOrder(e.target.value)} className="w-full p-3 border border-slate-300 rounded-lg bg-white focus:ring-2 focus:ring-blue-500">
                    <option value="desc">降順</option>
                    <option value="asc">昇順</option>
                  </select>
                </div>
                <div className="flex items-end">
                  <button onClick={exportResults} className="w-full bg-blue-600 text-white py-3 px-4 rounded-lg hover:bg-blue-700 transition-colors font-medium flex items-center justify-center">
                    <Download className="mr-2 w-4 h-4" />CSV出力
                  </button>
                </div>
              </div>

              <div className="overflow-x-auto bg-slate-50 rounded-xl border border-slate-200">
                <table className="w-full">
                  <thead className="bg-slate-100 border-b border-slate-200">
                    <tr>
                      <th className="px-4 py-4 text-left text-sm font-semibold text-slate-700">ティッカー</th>
                      <th className="px-4 py-4 text-left text-sm font-semibold text-slate-700">銘柄名</th>
                      <th className="px-4 py-4 text-right text-sm font-semibold text-slate-700">適性</th>
                      <th className="px-4 py-4 text-center text-sm font-semibold text-slate-700">アクション</th>
                      <th className="px-4 py-4 text-right text-sm font-semibold text-slate-700">価格</th>
                      <th className="px-4 py-4 text-right text-sm font-semibold text-slate-700">変化率</th>
                      <th className="px-4 py-4 text-right text-sm font-semibold text-slate-700">信頼度</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredResults.map((result, index) => {
                      const actionStyle = getActionStyle(result.action);
                      return (
                        <tr key={index} className="border-b border-slate-100 hover:bg-white transition-colors">
                          <td className="px-4 py-4 text-sm font-mono font-medium text-slate-800">{result.ticker}</td>
                          <td className="px-4 py-4 text-sm text-slate-800 max-w-48 truncate">{result.name}</td>
                          <td className="px-4 py-4 text-sm text-right font-semibold text-slate-800">{result.fitness_score.toFixed(1)}</td>
                          <td className="px-4 py-4 text-center">
                            <span className={`inline-flex items-center px-3 py-1.5 rounded-full text-xs font-semibold border ${actionStyle.color}`}>
                              {actionStyle.icon}<span className="ml-1.5">{result.action}</span>
                            </span>
                          </td>
                          <td className="px-4 py-4 text-sm text-right font-mono">
                            {result.action === 'ERROR' ? 'N/A' : `¥${result.price.toLocaleString()}`}
                          </td>
                          <td className={`px-4 py-4 text-sm text-right font-semibold ${
                            result.action === 'ERROR' ? 'text-gray-400' :
                            result.change_pct > 0 ? 'text-green-600' : 'text-red-600'
                          }`}>
                            {result.action === 'ERROR' ? 'N/A' :
                             `${result.change_pct > 0 ? '+' : ''}${result.change_pct.toFixed(2)}%`}
                          </td>
                          <td className="px-4 py-4 text-sm text-right font-semibold text-slate-800">
                            {result.action === 'ERROR' ? 'N/A' : `${(result.confidence * 100).toFixed(0)}%`}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>

              {summary.errorCount > 0 && (
                <div className="mt-4 bg-orange-50 border border-orange-200 rounded-lg p-4">
                  <p className="text-orange-800 font-semibold">⚠️ 一部銘柄の分析に失敗しました</p>
                  <p className="text-orange-600 text-sm mt-1">
                    データ取得失敗やテクニカル指標計算エラーが原因の可能性があります。<br />
                    時間をおいて再度実行するか、対象銘柄を絞り込んでお試しください。
                  </p>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

function App() {
  return (
    <div className="App">
      <StockAnalysisDashboard />
    </div>
  );
}

export default App;
