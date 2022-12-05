import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    def __init__(self):
        self.raw_matches_data: pd.DataFrame = pd.DataFrame()
        self.raw_players_data: pd.DataFrame = pd.DataFrame()

        self.players_data: pd.DataFrame = pd.DataFrame()
        self.final_players_data: pd.DataFrame = pd.DataFrame()
        self.common_data: pd.DataFrame = pd.DataFrame()

        self.scaler = StandardScaler()

        self._prefixes_list: list[str] = [f'p{i}_' for i in range(1, 6)]
        self._stats_features: list[str] = list()
        self._unique_players: list[int] = list()

    def preprocess(self,
                   matches_data: pd.DataFrame,
                   players_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare raw data to fitting ML models
            params:
            matches_data: pd.DataFrame - raw data with train matches info
            players_data: pd.DataFrame - raw data with players stats for each card

            return:
            final_data: pd.DataFrame - preprocessed and cleaned data with additional features
        """
        self.raw_matches_data = matches_data
        self.raw_players_data = players_data

        self._get_stats_features()

        self.players_data = self._transform_players_data()
        self._scale_players_data()
        self.final_players_data = self._inverse_transform_players_data()
        self.common_data = self.merge_players_with_matches(self.raw_matches_data)
        self.common_data = self.common_data.drop(
            columns=[f"p{i}_id_{side}" for i in range(1, 6) for side in ['first', 'second']])
        self.common_data = self.add_features(self.common_data)

        return self.common_data

    def _get_stats_features(self) -> None:
        """
        Get list of all statistical features

        return: None
        """
        self._stats_features = ['_'.join(feature.split('_')[1:]) for feature in
                                self.raw_players_data.iloc[0:1, 1:25].columns.tolist()]

    def _get_unique_players(self) -> None:
        """
        Get list of all unique players ids

        return: None
        """
        self._unique_players = list(set(self.raw_players_data.filter(regex=r'p\d_id', axis=1).values.flatten()))

    def _scale_players_data(self) -> None:
        """
        Scale players stats features with StandardScaler

        return: None
        """
        self.players_data.loc[:, self._stats_features] = self.scaler.fit_transform(self.players_data.iloc[:, 4:])

    def merge_players_with_matches(self, matches_data: pd.DataFrame) -> pd.DataFrame:
        """
        Merge players stats to matches data

        return: pd.DataFrame - merged data
        """
        common_data = matches_data.merge(self.final_players_data, left_on=['map_id', 'team1_id', 'map_name'],
                                         right_on=['map_id', 'team_id', 'map_name'])

        self.common_data = common_data.merge(self.final_players_data,
                                             left_on=['map_id', 'team2_id', 'map_name'],
                                             right_on=['map_id', 'team_id', 'map_name'],
                                             suffixes=['_first', '_second'])
        self.common_data = self.common_data.drop(columns=['team_id_first', 'team_id_second'])

        return self.common_data

    def add_features(self, common_data: pd.DataFrame) -> pd.DataFrame:
        """
        Compare players from both teams to each other by all stats

        """

        for feature in self._stats_features:
            for player_pos_first in [f"p{i}" for i in range(1, 6)]:
                for player_pos_second in [f"p{i}" for i in range(1, 6)]:
                    common_data.loc[:, f"{player_pos_first}_vs_{player_pos_second}_{feature}"] \
                        = common_data[f"{player_pos_first}_{feature}_first"] - common_data[
                        f"{player_pos_second}_{feature}_second"]
        return common_data

    def _transform_players_data(self) -> pd.DataFrame:
        """
        Transform raw players info to new view: each row describe only one player from current match
        return: pd.DataFrame - transformed players info DataFrame
        """

        all_players_stats_list = list()

        for row_index in range(self.raw_players_data.shape[0]):
            map_id = self.raw_players_data.loc[row_index, 'map_id']
            map_name = self.raw_players_data.loc[row_index, 'map_name']
            team_id = self.raw_players_data.loc[row_index, 'team_id']

            for prefix in self._prefixes_list:
                players_data_transformed = pd.DataFrame(
                    columns=['player_id', 'team_id', 'map_id', 'map_name'] + self._stats_features)
                players_data_transformed.loc[0, 'player_id'] = self.raw_players_data.loc[row_index, f"{prefix}id"]
                players_data_transformed.loc[0, 'map_id'] = map_id
                players_data_transformed.loc[0, 'map_name'] = map_name
                players_data_transformed.loc[0, 'team_id'] = team_id

                for feature in self._stats_features:
                    players_data_transformed.loc[0, feature] = self.raw_players_data.loc[
                        row_index, f"{prefix}{feature}"]
                all_players_stats_list.append(players_data_transformed)

        players_data_transformed = pd.concat(all_players_stats_list, ignore_index=True)

        return players_data_transformed

    def _inverse_transform_players_data(self) -> pd.DataFrame:
        players_matches = list()
        matches_groups = self.players_data.groupby(['map_id', 'team_id', 'map_name'])

        for match in matches_groups:
            map_id, team_id, map_name = match[0]
            players_data_current = match[1].drop(columns=['map_id', 'team_id', 'map_name']).values.flatten().tolist()
            players_data_current.append(team_id)
            players_data_current.append(map_name)
            players_data_current.append(map_id)
            current_match = pd.DataFrame()
            current_match.loc[0, self.raw_players_data.columns.tolist()] = players_data_current

            players_matches.append(current_match)

        players_matches = pd.concat(players_matches, ignore_index=True)
        players_matches['map_id'] = players_matches['map_id'].astype(int)
        players_matches['team_id'] = players_matches['team_id'].astype(int)
        players_matches['team_id'] = players_matches['team_id'].astype('category')
        for i in range(1, 6):
            players_matches[f"p{i}_id"] = players_matches[f"p{i}_id"].astype(int)
            players_matches[f"p{i}_id"] = players_matches[f"p{i}_id"].astype('category')

        return players_matches
